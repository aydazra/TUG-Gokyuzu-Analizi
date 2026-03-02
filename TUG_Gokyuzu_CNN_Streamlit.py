# -*- coding: utf-8 -*-
"""
TUG – Gökyüzü CNN Analizi (Streamlit v6 - FIXED)
Türkiye Ulusal Gözlemevleri

✅ DÜZELTMELER:
- TypeError: unsupported operand type(s) for *: 'NoneType' and 'int' → ÇÖZÜLDÜ
- tahmin_yap() fonksiyonu düzeltildi (gece tespitinde return değeri eksikti)
- Hata yönetimi eklendi
- Kullanıcı dostu mesajlar eklendi
"""

import cv2
import numpy as np
import os, re
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
from pathlib import Path
from datetime import datetime, date, time
import json
import warnings
warnings.filterwarnings('ignore')

# PIL/Pillow EXIF için
try:
    from PIL import Image
    from PIL.ExifTags import TAGS
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False

# TensorFlow import
try:
    import tensorflow as tf
    from tensorflow import keras
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False
    st.error("⚠️ TensorFlow yüklü değil! `pip install tensorflow` çalıştırın.")

# ═══════════════════════════════════════════════════════════
# STREAMLIT SAYFA AYARLARI
# ═══════════════════════════════════════════════════════════
st.set_page_config(
    page_title="TUG Gökyüzü CNN Analizi",
    page_icon="🔭",
    layout="wide"
)

st.markdown("""
<style>
    .main { background-color: #0f172a; }
    .stApp { background-color: #0f172a; color: #e2e8f0; }
    .metric-card {
        background: #1e293b;
        border: 1px solid #334155;
        border-radius: 10px;
        padding: 1rem;
        text-align: center;
    }
    h1, h2, h3, h4 { color: #e2e8f0 !important; }
    .stSelectbox label, .stTextInput label { color: #94a3b8 !important; }
    .prediction-box {
        background: #1e293b;
        border: 3px solid;
        border-radius: 12px;
        padding: 2rem;
        text-align: center;
        font-size: 1.5rem;
        font-weight: bold;
        margin: 1rem 0;
    }
    .acik { border-color: #10b981; color: #10b981; }
    .parcali { border-color: #f59e0b; color: #f59e0b; }
    .kapali { border-color: #ef4444; color: #ef4444; }
    .gece { border-color: #6366f1; color: #6366f1; }
</style>
""", unsafe_allow_html=True)

plt.rcParams.update({
    'figure.facecolor': '#0f172a',
    'axes.facecolor':   '#1e293b',
    'text.color':       '#e2e8f0',
    'axes.labelcolor':  '#94a3b8',
    'xtick.color':      '#64748b',
    'ytick.color':      '#64748b',
    'axes.edgecolor':   '#334155',
    'grid.color':       '#334155',
    'font.family':      'monospace',
})

# ═══════════════════════════════════════════════════════════
# GLOBAL DEĞİŞKENLER
# ═══════════════════════════════════════════════════════════

SINIF_ISIMLERI = ['🟢 Tam Açık', '🟡 Parçalı', '🔴 Kapalı']
SINIF_KODLARI = ['acik', 'parcali', 'kapali']
SINIF_RENKLER = ['#10b981', '#f59e0b', '#ef4444']
UZANTILAR = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif', '.fits'}

# ═══════════════════════════════════════════════════════════
# YARDIMCI FONKSİYONLAR - EXIF & ZAMAN
# ═══════════════════════════════════════════════════════════

def exif_saat_oku(img_path):
    """
    EXIF verilerinden saat bilgisini okur
    
    Returns:
        hour (int): Saat (0-23) veya None
    """
    if not PIL_AVAILABLE:
        return None
    
    try:
        img = Image.open(img_path)
        exif_data = img._getexif()
        
        if not exif_data:
            return None
        
        # DateTimeOriginal tag'ini bul (306 veya 36867)
        for tag_id, value in exif_data.items():
            tag = TAGS.get(tag_id, tag_id)
            
            if tag in ['DateTime', 'DateTimeOriginal', 'DateTimeDigitized']:
                # Format: "2025:08:15 14:07:23"
                try:
                    dt_str = str(value)
                    # Saat kısmını al
                    time_part = dt_str.split()[1]  # "14:07:23"
                    hour = int(time_part.split(':')[0])
                    return hour
                except:
                    continue
        
        return None
    except:
        return None


def dosya_adından_saat_oku(filename):
    """
    Dosya adından saat bilgisini çıkarmaya çalışır
    Örnekler:
    - "2018_09_15__14_07_23.jpg" → 14
    - "IMG_20250815_140723.jpg" → 14
    - "sky_2025-08-15_14-07.jpg" → 14
    
    Returns:
        hour (int): Saat (0-23) veya None
    """
    try:
        # Alt çizgi veya tire ile ayrılmış formatlar
        patterns = [
            r'_(\d{2})_\d{2}_\d{2}',  # _14_07_23
            r'-(\d{2})-\d{2}-\d{2}',  # -14-07-23
            r'_(\d{2})\d{2}\d{2}',    # _140723
            r'T(\d{2})\d{2}\d{2}',    # T140723 (ISO format)
        ]
        
        for pattern in patterns:
            match = re.search(pattern, filename)
            if match:
                hour = int(match.group(1))
                if 0 <= hour <= 23:
                    return hour
        
        return None
    except:
        return None


def gece_mi_tespit_et(img_path=None, hour=None, mean_brightness=None, latitude=39.0, longitude=32.0):
    """
    Görüntünün gece çekimi olup olmadığını tespit eder
    
    Öncelik sırası:
    1. EXIF saat bilgisi (en güvenilir)
    2. Dosya adından saat bilgisi
    3. Parlaklık analizi (fallback)
    
    Args:
        img_path: Görüntü dosya yolu
        hour: Manuel saat bilgisi (0-23)
        mean_brightness: Ortalama parlaklık (0-1)
        latitude, longitude: Konum (güneş açısı hesabı için - gelecekte)
    
    Returns:
        is_night (bool): Gece mi?
        detection_method (str): Tespit yöntemi
    """
    
    # 1. Manuel saat verildiyse
    if hour is not None:
        is_night = not (6 <= hour <= 18)
        return is_night, f"manuel_saat_{hour}"
    
    # 2. EXIF'ten saat oku
    if img_path and PIL_AVAILABLE:
        exif_hour = exif_saat_oku(img_path)
        if exif_hour is not None:
            is_night = not (6 <= exif_hour <= 18)
            return is_night, f"exif_saat_{exif_hour}"
    
    # 3. Dosya adından saat oku
    if img_path:
        filename = Path(img_path).name if isinstance(img_path, (str, Path)) else str(img_path)
        file_hour = dosya_adından_saat_oku(filename)
        if file_hour is not None:
            is_night = not (6 <= file_hour <= 18)
            return is_night, f"dosya_saat_{file_hour}"
    
    # 4. Parlaklık analizi (fallback)
    if mean_brightness is not None:
        is_night = mean_brightness < 0.15
        return is_night, f"parlaklik_{mean_brightness:.3f}"
    
    # Hiçbir bilgi yoksa False döndür (gündüz varsay)
    return False, "bilinmiyor"


# ═══════════════════════════════════════════════════════════
# MODEL YÜKLEME
# ═══════════════════════════════════════════════════════════

@st.cache_resource
def model_yukle(model_path):
    """CNN modelini yükler"""
    if not TF_AVAILABLE:
        return None
    if not os.path.exists(model_path):
        return None
    try:
        model = keras.models.load_model(model_path)
        return model
    except Exception as e:
        st.error(f"Model yükleme hatası: {e}")
        return None


@st.cache_data
def history_yukle(history_path):
    """Eğitim history'sini yükler"""
    if not os.path.exists(history_path):
        return None
    try:
        with open(history_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        st.error(f"History yükleme hatası: {e}")
        return None


# ═══════════════════════════════════════════════════════════
# GÖRÜNTÜ İŞLEME
# ═══════════════════════════════════════════════════════════

def roi_maske_olustur(h, w, kenar_yuzde=8):
    """Dairesel ROI maskesi"""
    maske = np.zeros((h, w), dtype=np.uint8)
    cx, cy = w // 2, h // 2
    r = int(min(cx, cy) * (1.0 - kenar_yuzde / 100.0))
    cv2.circle(maske, (cx, cy), r, 255, -1)
    return maske


def gorsel_hazirla(img, target_size=(128, 128)):
    """Görseli CNN için hazırlar"""
    h, w = img.shape[:2]
    
    # ROI maskesi uygula
    roi = roi_maske_olustur(h, w)
    roi_3ch = cv2.merge([roi, roi, roi])
    img = cv2.bitwise_and(img, roi_3ch)
    
    # Resize
    img = cv2.resize(img, target_size)
    
    # RGB ve normalize
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype(np.float32) / 255.0
    
    return img


def tahmin_yap(model, img, img_path=None, manual_hour=None):
    """
    CNN ile tahmin yapar - GECE ve GÜNDÜZ görüntüleri için
    
    ✅ YENİ: EXIF/dosya adı/parlaklık ile akıllı gece tespiti
    
    Args:
        model: TensorFlow model
        img: İşlenmiş görüntü array
        img_path: Orijinal dosya yolu (EXIF okuma için)
        manual_hour: Manuel saat (0-23)
    
    Returns:
        predicted_class: int (0, 1, 2)
        probabilities: numpy array
        confidence: float
        is_night: bool (gece görüntüsü mü?)
        detection_info: str (tespit yöntemi bilgisi)
    """
    
    # Ortalama parlaklık
    mean_brightness = np.mean(img)
    
    # ✅ AKILLI GECE TESPİTİ (EXIF → Dosya adı → Parlaklık)
    is_night, detection_method = gece_mi_tespit_et(
        img_path=img_path,
        hour=manual_hour,
        mean_brightness=mean_brightness
    )
    
    detection_info = f"Tespit: {detection_method}"
    
    # Batch dimension ekle
    img_batch = np.expand_dims(img, axis=0)
    
    # Tahmin (hem gece hem gündüz için)
    predictions = model.predict(img_batch, verbose=0)
    probabilities = predictions[0]
    
    predicted_class = int(np.argmax(probabilities))
    confidence = float(probabilities[predicted_class])
    
    return predicted_class, probabilities, confidence, is_night, detection_info


# ═══════════════════════════════════════════════════════════
# GÖRSELLEŞTIRME
# ═══════════════════════════════════════════════════════════

def tahmin_gorseli_olustur(original_img, predicted_class, probabilities, confidence, is_night=False):
    """
    Tahmin sonucunu görselleştirir
    
    ✅ YENİ: is_night parametresi eklendi - gece uyarısı gösterir
    """
    
    # ✅ HATA KONTROLÜ: probabilities None ise görselleştirme yapma
    if probabilities is None:
        st.error("⚠️ Tahmin olasılıkları hesaplanamadı")
        return None
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.patch.set_facecolor('#0f172a')
    
    # Orijinal görüntü
    ax1 = axes[0]
    ax1.imshow(cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB))
    
    # ✅ YENİ: Gece uyarısı
    if is_night:
        ax1.set_title('⚠️ GECE GÖRÜNTÜSÜ - Model gündüz için eğitildi', 
                     color='#f59e0b', fontsize=13, fontweight='bold')
    else:
        ax1.set_title('Orijinal Görüntü', color='#e2e8f0', fontsize=13, fontweight='bold')
    ax1.axis('off')
    
    # Tahmin bar chart
    ax2 = axes[1]
    ax2.set_facecolor('#1e293b')
    
    bars = ax2.barh(SINIF_ISIMLERI, probabilities * 100, 
                    color=SINIF_RENKLER, alpha=0.7, height=0.6)
    
    # Tahmin edilen sınıfı vurgula
    bars[predicted_class].set_alpha(1.0)
    bars[predicted_class].set_edgecolor('#e2e8f0')
    bars[predicted_class].set_linewidth(3)
    
    ax2.set_xlabel('Olasılık (%)', color='#94a3b8', fontsize=11)
    
    # ✅ YENİ: Başlıkta gece uyarısı
    title = f'CNN Tahmini: {SINIF_ISIMLERI[predicted_class]}\nGüven: {confidence*100:.1f}%'
    if is_night:
        title += '\n⚠️ Gece - Tahmin güvenilir olmayabilir'
    
    ax2.set_title(title, color='#e2e8f0', fontsize=13, fontweight='bold')
    ax2.set_xlim(0, 100)
    ax2.tick_params(colors='#64748b')
    ax2.grid(axis='x', color='#334155', alpha=0.3)
    
    # Değerleri bar üzerine yaz
    for i, (bar, prob) in enumerate(zip(bars, probabilities)):
        width = bar.get_width()
        ax2.text(width + 2, bar.get_y() + bar.get_height()/2, 
                f'{prob*100:.1f}%', 
                va='center', color='#e2e8f0', fontsize=10, fontweight='bold')
    
    plt.tight_layout()
    return fig


def egitim_grafikleri_goster(history_dict):
    """Eğitim history grafiklerini gösterir"""
    if not history_dict:
        return None
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    fig.patch.set_facecolor('#0f172a')
    
    epochs_range = range(1, len(history_dict['accuracy']) + 1)
    
    # Accuracy
    ax1.set_facecolor('#1e293b')
    ax1.plot(epochs_range, [x*100 for x in history_dict['accuracy']], 
             'o-', label='Training', color='#10b981', linewidth=2.5, markersize=7)
    ax1.plot(epochs_range, [x*100 for x in history_dict['val_accuracy']], 
             's-', label='Validation', color='#f59e0b', linewidth=2.5, markersize=7)
    ax1.set_xlabel('Epoch', color='#94a3b8', fontsize=12)
    ax1.set_ylabel('Accuracy (%)', color='#94a3b8', fontsize=12)
    ax1.set_title('Model Accuracy', color='#e2e8f0', fontsize=14, fontweight='bold')
    ax1.legend(facecolor='#1e293b', labelcolor='#94a3b8', fontsize=11, loc='lower right')
    ax1.grid(True, color='#334155', alpha=0.3, linestyle='--')
    ax1.tick_params(colors='#64748b')
    
    # Loss
    ax2.set_facecolor('#1e293b')
    ax2.plot(epochs_range, history_dict['loss'], 
             'o-', label='Training', color='#10b981', linewidth=2.5, markersize=7)
    ax2.plot(epochs_range, history_dict['val_loss'], 
             's-', label='Validation', color='#f59e0b', linewidth=2.5, markersize=7)
    ax2.set_xlabel('Epoch', color='#94a3b8', fontsize=12)
    ax2.set_ylabel('Loss', color='#94a3b8', fontsize=12)
    ax2.set_title('Model Loss', color='#e2e8f0', fontsize=14, fontweight='bold')
    ax2.legend(facecolor='#1e293b', labelcolor='#94a3b8', fontsize=11, loc='upper right')
    ax2.grid(True, color='#334155', alpha=0.3, linestyle='--')
    ax2.tick_params(colors='#64748b')
    
    plt.tight_layout()
    return fig


# ═══════════════════════════════════════════════════════════
# ANA UYGULAMA
# ═══════════════════════════════════════════════════════════

st.title("🔭 TUG Gökyüzü CNN Analizi")
st.markdown("**Türkiye Ulusal Gözlemevleri** - TensorFlow/Keras CNN ile Gözlem Kalitesi Analizi")
st.divider()

# ─── SIDEBAR ───
with st.sidebar:
    st.header("⚙️ Ayarlar")
    
    st.subheader("📁 Model Dosyaları")
    model_path = st.text_input(
        "Model (.h5):",
        value="tug_gokyuzu_cnn_model.h5",
        help="Eğitilmiş CNN modeli"
    )
    
    history_path = st.text_input(
        "Training History (.json):",
        value="training_history.json",
        help="Eğitim geçmişi JSON dosyası"
    )
    
    st.divider()
    
    st.subheader("🎯 Analiz Modu")
    analiz_modu = st.radio(
        "Mod Seçin:",
        ["📸 Tek Görüntü", "📁 Klasör Analizi", "📊 Model Performansı"],
        help="Yapmak istediğiniz analiz türünü seçin"
    )
    
    st.divider()
    
    # Model boyutu
    if os.path.exists(history_path):
        history_data = history_yukle(history_path)
        if history_data and 'img_size' in history_data:
            IMG_SIZE = tuple(history_data['img_size'])
            st.caption(f"ℹ️ Model görüntü boyutu: {IMG_SIZE}")
        else:
            IMG_SIZE = (128, 128)
            st.caption(f"ℹ️ Varsayılan boyut: {IMG_SIZE}")
    else:
        IMG_SIZE = (128, 128)
        st.caption(f"ℹ️ Varsayılan boyut: {IMG_SIZE}")

# ─── MODEL YÜKLEME ───
if not TF_AVAILABLE:
    st.error("❌ TensorFlow yüklü değil! Lütfen `pip install tensorflow` komutunu çalıştırın.")
    st.stop()

model = model_yukle(model_path)
history_dict = history_yukle(history_path)

if model is None:
    st.error(f"❌ Model yüklenemedi: `{model_path}`")
    st.info("💡 **Adımlar:**\n1. `TUG_Gokyuzu_v6_CNN_Training.py` scriptini çalıştırın\n2. Model dosyası oluşturulduktan sonra bu uygulamayı tekrar başlatın")
    st.stop()

st.success(f"✅ Model başarıyla yüklendi: `{model_path}`")

# Model bilgileri
if history_dict:
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("📚 Toplam Epoch", history_dict.get('epochs', 'N/A'))
    col2.metric("🎯 Final Train Acc", f"{history_dict.get('final_train_acc', 0)*100:.2f}%")
    col3.metric("✅ Final Val Acc", f"{history_dict.get('final_val_acc', 0)*100:.2f}%")
    col4.metric("🏆 Best Val Acc", f"{history_dict.get('best_val_acc', 0)*100:.2f}%")

st.divider()

# ═══════════════════════════════════════════════════════════
# MOD 1: TEK GÖRÜNTÜ
# ═══════════════════════════════════════════════════════════

if analiz_modu == "📸 Tek Görüntü":
    st.subheader("📸 Tek Görüntü Analizi")
    st.caption("All-sky kamera görüntüsü yükleyin ve CNN ile sınıflandırın")
    
    uploaded_file = st.file_uploader(
        "Görüntü seçin:",
        type=['jpg', 'jpeg', 'png', 'bmp', 'tiff', 'tif'],
        help="Gökyüzü fotoğrafı (.jpg, .png, vb.)"
    )
    
    if uploaded_file is not None:
        # Dosyayı geçici kaydet (EXIF okuma için)
        import tempfile
        with tempfile.NamedTemporaryFile(delete=False, suffix=uploaded_file.name) as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            tmp_path = tmp_file.name
        
        # Dosyayı oku
        file_bytes = np.asarray(bytearray(uploaded_file.getvalue()), dtype=np.uint8)
        original_img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        
        if original_img is None:
            st.error("❌ Görüntü okunamadı!")
        else:
            # Görseli hazırla
            processed_img = gorsel_hazirla(original_img, target_size=IMG_SIZE)
            
            # ✅ Manuel saat girişi (opsiyonel)
            with st.expander("⏰ Manuel Saat Girişi (EXIF yoksa)", expanded=False):
                col_h1, col_h2 = st.columns(2)
                with col_h1:
                    manuel_saat_aktif = st.checkbox("Manuel saat kullan", value=False)
                with col_h2:
                    if manuel_saat_aktif:
                        manuel_saat = st.slider("Saat", 0, 23, 12)
                    else:
                        manuel_saat = None
            
            # Tahmin yap (EXIF/dosya adı/parlaklık ile gece tespiti)
            with st.spinner('🔮 CNN tahmini yapılıyor...'):
                predicted_class, probabilities, confidence, is_night, detection_info = tahmin_yap(
                    model, processed_img, 
                    img_path=tmp_path,
                    manual_hour=manuel_saat if manuel_saat_aktif else None
                )
            
            # Geçici dosyayı sil
            try:
                os.unlink(tmp_path)
            except:
                pass
            
            # ✅ GECE UYARISI (ama yine de tahmin göster!)
            if is_night:
                st.warning(f"🌙 **Gece Görüntüsü Tespit Edildi!** ({detection_info}) Model gündüz görüntüleri için eğitilmiştir. Tahmin daha az güvenilir olabilir.")
            else:
                st.info(f"☀️ **Gündüz Görüntüsü** ({detection_info})")
            
            # Tahmin sonucunu göster
            durum_class = SINIF_KODLARI[predicted_class]
            
            # ✅ YENİ: Gece için farklı renk tonu
            if is_night:
                st.markdown(f"""
                <div class="prediction-box {durum_class}" style="border-color: #6366f1; opacity: 0.9;">
                    🌙 {SINIF_ISIMLERI[predicted_class]}<br>
                    <span style="font-size: 1rem;">Güven: {confidence*100:.2f}% (Gece Modu)</span>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="prediction-box {durum_class}">
                    {SINIF_ISIMLERI[predicted_class]}<br>
                    <span style="font-size: 1rem;">Güven: {confidence*100:.2f}%</span>
                </div>
                """, unsafe_allow_html=True)
            
            # Görselleştirme
            fig = tahmin_gorseli_olustur(original_img, predicted_class, probabilities, confidence, is_night)
            
            if fig is not None:
                st.pyplot(fig)
                plt.close(fig)
            
            # Detaylı olasılıklar
            with st.expander("📊 Detaylı Olasılık Dağılımı"):
                prob_df = pd.DataFrame({
                    'Sınıf': SINIF_ISIMLERI,
                    'Olasılık': [f"{p*100:.2f}%" for p in probabilities],
                    'Raw Score': [f"{p:.6f}" for p in probabilities]
                })
                st.dataframe(prob_df, use_container_width=True, hide_index=True)
            
            # Açıklama
            if is_night:
                st.info(f"""
                **ℹ️ Gece Modu CNN Tahmini:**
                - Model gündüz görüntüleri ile eğitilmiştir, ancak gece bulut örtüsü de analiz edilebilir
                - Gece görüntülerinde yıldızlar, ay ve bulutlar farklı parlaklık dağılımı gösterir
                - Güven skoru: %{confidence*100:.1f} → Bu değer gece için {"yüksek ancak dikkatli yorumlanmalı" if confidence > 0.7 else "düşük, dikkatle değerlendirin"}
                - **Öneri:** Gece gözlemleri için özel eğitilmiş bir model kullanmak daha iyi sonuç verir
                """)
            else:
                st.info(f"""
                **ℹ️ CNN Tahmin Açıklaması:**
                - Model, görüntüyü 3 sınıfa ayırıyor: Tam Açık, Parçalı, Kapalı
                - En yüksek olasılığa sahip sınıf seçiliyor
                - Güven skoru, modelin tahmininden ne kadar emin olduğunu gösterir
                - %{confidence*100:.1f} güven → Model bu tahmininden {"çok" if confidence > 0.8 else "oldukça" if confidence > 0.6 else "nispeten"} emin
                """)

# ═══════════════════════════════════════════════════════════
# MOD 2: KLASÖR ANALİZİ
# ═══════════════════════════════════════════════════════════

elif analiz_modu == "📁 Klasör Analizi":
    st.subheader("📁 Klasör (Batch) Analizi")
    st.caption("Bir klasördeki tüm görüntüleri toplu olarak analiz edin")
    
    klasor_yolu = st.text_input(
        "Klasör Yolu:",
        value="/path/to/images",
        help="Analiz edilecek gökyüzü görüntülerinin bulunduğu klasör"
    )
    
    col_btn1, col_btn2 = st.columns([3, 1])
    with col_btn1:
        analiz_btn = st.button("🚀 Analizi Başlat", use_container_width=True, type="primary")
    with col_btn2:
        max_files = st.number_input("Max dosya:", min_value=10, max_value=10000, value=500, step=50)
    
    if analiz_btn:
        klasor = Path(klasor_yolu)
        
        if not klasor.exists():
            st.error(f"❌ Klasör bulunamadı: `{klasor_yolu}`")
        else:
            # Dosyaları bul
            dosyalar = []
            for ext in UZANTILAR:
                dosyalar.extend(list(klasor.rglob(f"*{ext}")))
            
            dosyalar = dosyalar[:max_files]  # Limit
            
            if not dosyalar:
                st.warning("⚠️ Klasörde görüntü bulunamadı!")
            else:
                st.info(f"📂 {len(dosyalar)} görüntü bulundu. Analiz başlatılıyor...")
                
                kayitlar = []
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                gece_sayisi = 0
                detection_methods = {}
                
                for i, dosya in enumerate(dosyalar):
                    status_text.caption(f"🔍 İşleniyor: {dosya.name} ({i+1}/{len(dosyalar)})")
                    
                    img = cv2.imread(str(dosya))
                    if img is None:
                        continue
                    
                    processed = gorsel_hazirla(img, target_size=IMG_SIZE)
                    
                    # ✅ Dosya yolu ile tahmin (EXIF okuma için)
                    pred_class, probs, conf, is_night, detection_method = tahmin_yap(
                        model, processed, 
                        img_path=str(dosya)
                    )
                    
                    # ✅ GECE İŞARETLE AMA TAHMİN YAP
                    if is_night:
                        gece_sayisi += 1
                        # Tespit yöntemlerini say
                        method_key = detection_method.split('_')[0]
                        detection_methods[method_key] = detection_methods.get(method_key, 0) + 1
                    
                    kayitlar.append({
                        'dosya': dosya.name,
                        'yol': str(dosya.parent),
                        'durum': SINIF_ISIMLERI[pred_class].split()[1],  # Emoji'siz
                        'durum_kod': pred_class,
                        'gece': '🌙' if is_night else '',
                        'tespit': detection_method,
                        'guven': f"{conf*100:.2f}%",
                        'tam_acik': f"{probs[0]*100:.2f}%",
                        'parcali': f"{probs[1]*100:.2f}%",
                        'kapali': f"{probs[2]*100:.2f}%"
                    })
                    
                    progress_bar.progress((i + 1) / len(dosyalar))
                
                progress_bar.empty()
                status_text.empty()
                
                if not kayitlar:
                    st.error("❌ Hiç görüntü işlenemedi!")
                else:
                    df = pd.DataFrame(kayitlar)
                    
                    # İstatistikler
                    st.success(f"✅ {len(df)} görüntü başarıyla analiz edildi!")
                    
                    if gece_sayisi > 0:
                        method_info = ", ".join([f"{k}: {v}" for k, v in detection_methods.items()])
                        st.info(f"🌙 {gece_sayisi} gece görüntüsü tespit edildi (Yöntemler: {method_info})")
                    
                    n_acik = (df['durum_kod'] == 0).sum()
                    n_parcali = (df['durum_kod'] == 1).sum()
                    n_kapali = (df['durum_kod'] == 2).sum()
                    
                    col1, col2, col3, col4 = st.columns(4)
                    col1.metric("🟢 Tam Açık", f"{n_acik} ({n_acik/len(df)*100:.1f}%)")
                    col2.metric("🟡 Parçalı", f"{n_parcali} ({n_parcali/len(df)*100:.1f}%)")
                    col3.metric("🔴 Kapalı", f"{n_kapali} ({n_kapali/len(df)*100:.1f}%)")
                    col4.metric("🌙 Gece Görüntü", f"{gece_sayisi} ({gece_sayisi/len(df)*100:.1f}%)")
                    
                    # Grafikler
                    col_g1, col_g2 = st.columns(2)
                    
                    with col_g1:
                        # Pasta grafiği
                        fig1, ax1 = plt.subplots(figsize=(7, 7))
                        fig1.patch.set_facecolor('#0f172a')
                        ax1.set_facecolor('#1e293b')
                        
                        sizes = [n_acik, n_parcali, n_kapali]
                        labels = [f'Tam Açık\n{n_acik}', f'Parçalı\n{n_parcali}', f'Kapalı\n{n_kapali}']
                        explode = (0.05, 0, 0) if n_acik == max(sizes) else (0, 0.05, 0) if n_parcali == max(sizes) else (0, 0, 0.05)
                        
                        ax1.pie(sizes, labels=labels, colors=SINIF_RENKLER, autopct='%1.1f%%',
                               startangle=90, textprops={'color':'#e2e8f0', 'fontsize':11, 'fontweight':'bold'},
                               wedgeprops={'edgecolor':'#0f172a', 'linewidth':2}, explode=explode)
                        
                        title_text = 'Dağılım (Tüm Görüntüler)'
                        if gece_sayisi > 0:
                            title_text += f'\n🌙 {gece_sayisi} gece dahil'
                        ax1.set_title(title_text, color='#e2e8f0', fontsize=13, fontweight='bold', pad=20)
                        
                        st.pyplot(fig1)
                        plt.close(fig1)
                    
                    with col_g2:
                            # Bar chart
                            fig2, ax2 = plt.subplots(figsize=(7, 7))
                            fig2.patch.set_facecolor('#0f172a')
                            ax2.set_facecolor('#1e293b')
                            
                            bars = ax2.bar(['Tam Açık', 'Parçalı', 'Kapalı'], sizes, 
                                          color=SINIF_RENKLER, alpha=0.8, edgecolor='#e2e8f0', linewidth=2)
                            ax2.set_ylabel('Görüntü Sayısı', color='#94a3b8', fontsize=11)
                            ax2.set_title('Sınıf Dağılımı', color='#e2e8f0', fontsize=13, fontweight='bold')
                            ax2.tick_params(colors='#64748b')
                            ax2.grid(axis='y', color='#334155', alpha=0.3, linestyle='--')
                            
                            # Bar üzerine değer yaz
                            for bar, count in zip(bars, sizes):
                                height = bar.get_height()
                                ax2.text(bar.get_x() + bar.get_width()/2., height,
                                       f'{count}',
                                       ha='center', va='bottom', color='#e2e8f0', fontsize=12, fontweight='bold')
                            
                            st.pyplot(fig2)
                            plt.close(fig2)
                    
                    # Detay tablosu
                    st.subheader("🗂️ Detaylı Sonuçlar")
                    goster_df = df[['dosya', 'durum', 'gece', 'tespit', 'guven', 'tam_acik', 'parcali', 'kapali']]
                    st.dataframe(goster_df.rename(columns={
                        'dosya': 'Dosya Adı',
                        'durum': 'Durum',
                        'gece': 'Gece',
                        'tespit': 'Tespit Yöntemi',
                        'guven': 'Güven',
                        'tam_acik': 'Tam Açık %',
                        'parcali': 'Parçalı %',
                        'kapali': 'Kapalı %'
                    }), use_container_width=True, height=400)
                    
                    # CSV İndir
                    csv = goster_df.to_csv(index=False, encoding='utf-8-sig').encode('utf-8-sig')
                    st.download_button(
                        "📥 Sonuçları CSV Olarak İndir",
                        data=csv,
                        file_name=f"cnn_analiz_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        mime="text/csv",
                        use_container_width=True
                    )

# ═══════════════════════════════════════════════════════════
# MOD 3: MODEL PERFORMANSI
# ═══════════════════════════════════════════════════════════

elif analiz_modu == "📊 Model Performansı":
    st.subheader("📊 CNN Model Performansı ve Eğitim Geçmişi")
    
    if not history_dict:
        st.warning("⚠️ Eğitim history dosyası bulunamadı veya okunamadı.")
        st.info(f"💡 Beklenen dosya: `{history_path}`")
    else:
        # Özet metrikler
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### 📈 Eğitim Metrikleri")
            
            metric_col1, metric_col2 = st.columns(2)
            metric_col1.metric("Toplam Epoch", history_dict['epochs'])
            metric_col2.metric("Batch Size", history_dict.get('batch_size', 'N/A'))
            
            metric_col3, metric_col4 = st.columns(2)
            metric_col3.metric("Learning Rate", f"{history_dict.get('learning_rate', 0):.6f}")
            metric_col4.metric("Görüntü Boyutu", str(tuple(history_dict.get('img_size', [128, 128]))))
            
            st.divider()
            
            st.markdown("### 🎯 Accuracy Metrikleri")
            metric_col5, metric_col6 = st.columns(2)
            metric_col5.metric("Final Train Acc", f"{history_dict['final_train_acc']*100:.2f}%")
            metric_col6.metric("Final Val Acc", f"{history_dict['final_val_acc']*100:.2f}%")
            
            st.metric("🏆 Best Val Acc", f"{history_dict['best_val_acc']*100:.2f}%", 
                     delta=f"+{(history_dict['best_val_acc'] - history_dict['final_val_acc'])*100:.2f}%")
        
        with col2:
            st.markdown("### ℹ️ Model Bilgileri")
            st.code(f"""
Model Dosyası: {model_path}
Eğitim Tarihi: {history_dict.get('timestamp', 'N/A')}
Toplam Epoch: {history_dict['epochs']}
Best Validation Accuracy: {history_dict['best_val_acc']*100:.2f}%

Sınıflar:
- 0: Tam Açık (0-40% bulut)
- 1: Parçalı (40-70% bulut)
- 2: Kapalı (70-100% bulut)
            """, language="")
        
        st.divider()
        
        # Eğitim grafikleri
        st.markdown("### 📊 Eğitim Grafikleri")
        fig = egitim_grafikleri_goster(history_dict)
        if fig:
            st.pyplot(fig)
            plt.close(fig)
        
        # Epoch detayları
        with st.expander("📋 Epoch-by-Epoch Detayları", expanded=False):
            epoch_df = pd.DataFrame({
                'Epoch': range(1, len(history_dict['accuracy']) + 1),
                'Train Acc': [f"{x*100:.2f}%" for x in history_dict['accuracy']],
                'Val Acc': [f"{x*100:.2f}%" for x in history_dict['val_accuracy']],
                'Train Loss': [f"{x:.4f}" for x in history_dict['loss']],
                'Val Loss': [f"{x:.4f}" for x in history_dict['val_loss']]
            })
            st.dataframe(epoch_df, use_container_width=True, height=400)
            
            # CSV indir
            csv = epoch_df.to_csv(index=False, encoding='utf-8-sig').encode('utf-8-sig')
            st.download_button(
                "📥 Epoch Detaylarını İndir",
                data=csv,
                file_name="epoch_details.csv",
                mime="text/csv"
            )
        
        # Model mimarisi bilgisi
        with st.expander("🏗️ Model Mimarisi", expanded=False):
            st.markdown("""
            **CNN Mimari Özeti:**
            
            ```
            Input (128x128x3)
                ↓
            Conv Block 1 (32 filters)
                ↓
            Conv Block 2 (64 filters)
                ↓
            Conv Block 3 (128 filters)
                ↓
            Conv Block 4 (256 filters)
                ↓
            Global Average Pooling
                ↓
            Dense (512) + Dropout(0.5)
                ↓
            Dense (256) + Dropout(0.5)
                ↓
            Output (3) - Softmax
            ```
            
            **Her Conv Block:**
            - Conv2D (3x3, ReLU) + BatchNorm
            - Conv2D (3x3, ReLU) + BatchNorm
            - MaxPooling (2x2)
            - Dropout (0.25-0.3)
            
            **Toplam Parametreler:** ~2-3M (görüntü boyutuna göre değişir)
            """)

# ═══════════════════════════════════════════════════════════
# FOOTER
# ═══════════════════════════════════════════════════════════

st.divider()
st.caption("🔭 **TUG - Türkiye Ulusal Gözlemevleri** | CNN Tabanlı Gökyüzü Analizi v6.0 FIXED | TensorFlow/Keras")
st.caption("✅ TypeError hatası düzeltildi - Gece tespiti eklendi")