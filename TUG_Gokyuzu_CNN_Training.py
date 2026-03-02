# -*- coding: utf-8 -*-
"""
TUG – Gökyüzü CNN Model Eğitimi (TensorFlow)
Türkiye Ulusal Gözlemevleri

v6 - TensorFlow/Keras CNN Modeli:
- Epoch bazlı eğitim
- Accuracy & Loss tracking
- Model checkpoint
- Eğitim grafikleri
- Training history kaydetme
"""
from tensorflow.keras.callbacks import TensorBoard
import cv2
import numpy as np
import os, re
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime, date, time
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
from tensorflow import keras
from tensorflow.keras import layers, models, callbacks, regularizers
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
import json
import warnings
warnings.filterwarnings('ignore')

# GPU Ayarları
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f"✅ {len(gpus)} GPU bulundu")
    except RuntimeError as e:
        print(f"GPU ayarlama hatası: {e}")

# ═══════════════════════════════════════════════════════════
# AYARLAR
# ═══════════════════════════════════════════════════════════

# ⚠️ BURAYA VERİ KLASÖRÜNÜZÜ GİRİN
DATA_PATH = "C:\\Users\\Azra\\Desktop\\GokyuzuApp\\ASC_Data"

# Model Parametreleri
IMG_SIZE = (160, 160)  # CNN için görüntü boyutu (128x128 hızlı eğitim için)
BATCH_SIZE = 16
EPOCHS = 100
LEARNING_RATE = 0.003
VALIDATION_SPLIT = 0.2

# 🆕 DATA AUGMENTATION AYARLARI
USE_AUGMENTATION = False  # Data augmentation açık/kapalı
AUGMENTATION_STRENGTH = "medium"  # "light", "medium", "strong"

# 🆕 REGULARIZATION AYARLARI
L2_WEIGHT = 0.001  # L2 regularization weight
DROPOUT_CONV = 0.2  # Conv bloklarındaki dropout (0.25'ten artırıldı)
DROPOUT_DENSE = 0.3  # Dense katmanlardaki dropout

# Eşik değerleri (sınıflandırma için)
TAM_ACIK_ESIK = 40   # < %40 → Tam Açık
PARCALI_ESIK = 70    # < %70 → Parçalı

# Kaydetme yolları
MODEL_SAVE_PATH = "tug_gokyuzu_cnn_model.h5"
WEIGHTS_SAVE_PATH = "tug_gokyuzu_weights.h5"
HISTORY_SAVE_PATH = "training_history.json"

UZANTILAR = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif', '.fits'}

# ═══════════════════════════════════════════════════════════
# YARDIMCI FONKSİYONLAR
# ═══════════════════════════════════════════════════════════

def roi_maske_olustur(h, w, kenar_yuzde=8):
    """Dairesel ROI maskesi"""
    maske = np.zeros((h, w), dtype=np.uint8)
    cx, cy = w // 2, h // 2
    r = int(min(cx, cy) * (1.0 - kenar_yuzde / 100.0))
    cv2.circle(maske, (cx, cy), r, 255, -1)
    return maske


def disk_maskele(gray, roi, esik=248):
    """Merkezdeki parlak disk/güneş bölgesini maskeler"""
    h, w = gray.shape
    disk = np.zeros_like(gray)
    toplam_roi = max(int(np.sum(roi > 0)), 1)
    max_alan = toplam_roi * 0.06
    
    masked_gray = cv2.bitwise_and(gray, gray, mask=roi)
    _, parlak = cv2.threshold(masked_gray, esik, 255, cv2.THRESH_BINARY)
    
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    parlak = cv2.morphologyEx(parlak, cv2.MORPH_CLOSE, kernel)
    parlak = cv2.dilate(parlak, kernel, iterations=2)
    
    cnts, _ = cv2.findContours(parlak, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for c in cnts:
        alan = cv2.contourArea(c)
        if 50 < alan < max_alan:
            cv2.drawContours(disk, [c], -1, 255, -1)
    
    return disk


def basit_bulut_orani_hesapla(img_path):
    """
    Klasik algoritma ile bulut oranı hesaplar (etiketleme için)
    Returns: cloud_pct (0-100) veya None
    """
    img = cv2.imread(str(img_path))
    if img is None:
        return None
    
    h, w = img.shape[:2]
    if max(h, w) > 900:
        scale = 900 / max(h, w)
        img = cv2.resize(img, (int(w * scale), int(h * scale)))
        h, w = img.shape[:2]
    
    roi = roi_maske_olustur(h, w)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    disk = disk_maskele(gray, roi, esik=248)
    gecerli = cv2.bitwise_and(roi, cv2.bitwise_not(disk))
    
    # HSV Saturation tabanlı (gündüz için)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    s = hsv[:, :, 1]
    
    pix = gecerli > 0
    s_vals = s[pix]
    
    if len(s_vals) < 100:
        return None
    
    # Düşük saturation = bulut
    bulut_n = np.sum(s_vals < 35)
    cloud_pct = (bulut_n / len(s_vals)) * 100
    
    return float(np.clip(cloud_pct, 0, 100))


def bulut_oranini_sinifa_cevir(cloud_pct):
    """Bulut oranını sınıfa çevirir"""
    if cloud_pct < TAM_ACIK_ESIK:
        return 0  # Tam Açık
    elif cloud_pct < PARCALI_ESIK:
        return 1  # Parçalı
    else:
        return 2  # Kapalı


def gorsel_yukle_ve_hazirla(img_path, target_size=IMG_SIZE):
    """Görseli yükler, ROI uygular ve normalize eder"""
    img = cv2.imread(str(img_path))
    if img is None:
        return None
    
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


def klasorden_tarih(klasor_adi):
    """Klasör adından tarih çıkarır"""
    sayilar = re.findall(r'\d+', klasor_adi)
    for s in sayilar:
        if len(s) == 8:
            try:
                return date(int(s[:4]), int(s[4:6]), int(s[6:8]))
            except:
                pass
    if len(sayilar) >= 3:
        try:
            return date(int(sayilar[0]), int(sayilar[1]), int(sayilar[2]))
        except:
            pass
    return None


# ═══════════════════════════════════════════════════════════
# VERİ SETİ OLUŞTURMA
# ═══════════════════════════════════════════════════════════

def veri_seti_olustur(data_path, max_per_class=1000, verbose=True):
    """
    Klasördeki görüntüleri yükler ve otomatik etiketler
    
    Returns:
        X: numpy array (N, H, W, 3)
        y: numpy array (N,)
        dosya_listesi: list of paths
    """
    base = Path(data_path)
    if not base.exists():
        raise FileNotFoundError(f"Klasör bulunamadı: {data_path}")
    
    X_list = []
    y_list = []
    dosya_listesi = []
    class_counts = {0: 0, 1: 0, 2: 0}
    
    if verbose:
        print(f"\n{'='*60}")
        print(f"🔍 VERİ SETİ OLUŞTURULUYOR")
        print(f"{'='*60}")
        print(f"Klasör: {data_path}")
        print(f"Görüntü boyutu: {IMG_SIZE}")
        print(f"Her sınıftan max: {max_per_class if max_per_class else 'Sınırsız'}")
        print(f"{'='*60}\n")
    
    # Tüm klasörleri tara
    tarih_klasorleri = []
    for klasor in sorted(base.iterdir()):
        if klasor.is_dir():
            tarih = klasorden_tarih(klasor.name)
            if tarih:
                dosyalar = [f for f in klasor.iterdir() 
                           if f.is_file() and f.suffix.lower() in UZANTILAR]
                if dosyalar:
                    tarih_klasorleri.append((tarih, klasor, dosyalar))
    
    total_folders = len(tarih_klasorleri)
    
    for idx, (tarih, klasor, dosyalar) in enumerate(tarih_klasorleri, 1):
        if verbose:
            print(f"[{idx}/{total_folders}] 📁 {klasor.name} → {len(dosyalar)} dosya", end=" ")
        
        processed = 0
        for dosya in dosyalar:
            # Eğer sınıf limiti dolduysa geç
            if max_per_class:
                if all(class_counts[c] >= max_per_class for c in [0, 1, 2]):
                    break
            
            # Bulut oranı hesapla
            cloud_pct = basit_bulut_orani_hesapla(dosya)
            if cloud_pct is None:
                continue
            
            # Sınıfa çevir
            sinif = bulut_oranini_sinifa_cevir(cloud_pct)
            
            # Sınıf limiti kontrolü
            if max_per_class and class_counts[sinif] >= max_per_class:
                continue
            
            # Görseli yükle
            img = gorsel_yukle_ve_hazirla(dosya)
            if img is None:
                continue
            
            X_list.append(img)
            y_list.append(sinif)
            dosya_listesi.append(str(dosya))
            class_counts[sinif] += 1
            processed += 1
        
        if verbose:
            print(f"✓ {processed} işlendi [0:{class_counts[0]}, 1:{class_counts[1]}, 2:{class_counts[2]}]")
    
    if verbose:
        print(f"\n{'='*60}")
        print(f"📊 VERİ SETİ HAZIR")
        print(f"{'='*60}")
        print(f"🟢 Tam Açık: {class_counts[0]} görüntü")
        print(f"🟡 Parçalı: {class_counts[1]} görüntü")
        print(f"🔴 Kapalı: {class_counts[2]} görüntü")
        print(f"📊 Toplam: {len(X_list)} görüntü")
        print(f"{'='*60}\n")
    
    if len(X_list) == 0:
        raise ValueError("Hiç görüntü yüklenemedi!")
    
    X = np.array(X_list, dtype=np.float32)
    y = np.array(y_list, dtype=np.int32)
    
    return X, y, dosya_listesi


# ═══════════════════════════════════════════════════════════
# CNN MODELİ
# ═══════════════════════════════════════════════════════════
# Satır 188'den ÖNCE (cnn_model_olustur fonksiyonundan önce) ekleyin:

# ═══════════════════════════════════════════════════════════
# 🆕 DATA AUGMENTATION GENERATOR
# ═══════════════════════════════════════════════════════════

def get_augmentation_config(strength="medium"):
    """Augmentation ayarlarını döndürür"""
    configs = {
        "light": {
            'rotation_range': 10,
            'width_shift_range': 0.05,
            'height_shift_range': 0.05,
            'horizontal_flip': True,
            'vertical_flip': True,
            'zoom_range': 0.05,
            'brightness_range': [0.9, 1.1]
        },
        "medium": {
            'rotation_range': 20,
            'width_shift_range': 0.1,
            'height_shift_range': 0.1,
            'horizontal_flip': True,
            'vertical_flip': True,
            'zoom_range': 0.1,
            'brightness_range': [0.8, 1.2],
            'shear_range': 0.1
        },
        "strong": {
            'rotation_range': 30,
            'width_shift_range': 0.15,
            'height_shift_range': 0.15,
            'horizontal_flip': True,
            'vertical_flip': True,
            'zoom_range': 0.15,
            'brightness_range': [0.7, 1.3],
            'shear_range': 0.15,
            'fill_mode': 'reflect'
        }
    }
    return configs.get(strength, configs["medium"])

def cnn_model_olustur(input_shape=(128, 128, 3), num_classes=3):
    """
    Gökyüzü sınıflandırması için CNN modeli
    
    Mimari:
    - 4 Conv Blocks (32, 64, 128, 256)
    - BatchNorm + Dropout
    - Global Average Pooling
    - Dense layers
    - Softmax output
    """
    model = models.Sequential([
        # Input
        layers.Input(shape=input_shape),
        
        # Block 1: 32 filters
        layers.Conv2D(32, (3, 3), activation='relu', padding='same',
                     kernel_regularizer=regularizers.l2(L2_WEIGHT)),  # 🆕 L2 eklendi
        layers.BatchNormalization(),
        layers.Conv2D(32, (3, 3), activation='relu', padding='same',
                     kernel_regularizer=regularizers.l2(L2_WEIGHT)),  # 🆕 L2 eklendi
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(DROPOUT_CONV),  # 🆕 0.25'ten DROPOUT_CONV'a değişti
        
        # Block 2: 64 filters
        layers.Conv2D(64, (3, 3), activation='relu', padding='same',
                     kernel_regularizer=regularizers.l2(L2_WEIGHT)),  # 🆕
        layers.BatchNormalization(),
        layers.Conv2D(64, (3, 3), activation='relu', padding='same',
                     kernel_regularizer=regularizers.l2(L2_WEIGHT)),  # 🆕
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(DROPOUT_CONV),  # 🆕
        
        # Block 3: 128 filters
        layers.Conv2D(128, (3, 3), activation='relu', padding='same',
                     kernel_regularizer=regularizers.l2(L2_WEIGHT)),  # 🆕
        layers.BatchNormalization(),
        layers.Conv2D(128, (3, 3), activation='relu', padding='same',
                     kernel_regularizer=regularizers.l2(L2_WEIGHT)),  # 🆕
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(DROPOUT_CONV + 0.05),  # 🆕 Biraz daha yüksek
        
        # Block 4: 256 filters
        layers.Conv2D(256, (3, 3), activation='relu', padding='same',
                     kernel_regularizer=regularizers.l2(L2_WEIGHT)),  # 🆕
        layers.BatchNormalization(),
        layers.Conv2D(256, (3, 3), activation='relu', padding='same',
                     kernel_regularizer=regularizers.l2(L2_WEIGHT)),  # 🆕
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(DROPOUT_CONV + 0.05),  # 🆕
        
        # Global pooling
        layers.GlobalAveragePooling2D(),
        
        # Dense layers
        layers.Dense(512, activation='relu',
                    kernel_regularizer=regularizers.l2(L2_WEIGHT)),  # 🆕
        layers.BatchNormalization(),
        layers.Dropout(DROPOUT_DENSE),  # 🆕 0.5'ten DROPOUT_DENSE'e
        
        layers.Dense(256, activation='relu',
                    kernel_regularizer=regularizers.l2(L2_WEIGHT)),  # 🆕
        layers.BatchNormalization(),
        layers.Dropout(DROPOUT_DENSE),  # 🆕
        
        # Output
        layers.Dense(num_classes, activation='softmax', name='output')
    ])
    
    return model

# Satır 278'den ÖNCE (egitim_grafikleri_ciz fonksiyonundan önce) ekleyin:

# ═══════════════════════════════════════════════════════════
# 🆕 CONFUSION MATRIX
# ═══════════════════════════════════════════════════════════

def plot_confusion_matrix(y_true, y_pred, save_path='confusion_matrix.png'):
    """Confusion matrix çizer ve kaydeder"""
    cm = confusion_matrix(y_true, y_pred)
    
    fig, ax = plt.subplots(figsize=(10, 8))
    fig.patch.set_facecolor('#0f172a')
    ax.set_facecolor('#1e293b')
    
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Tam Açık', 'Parçalı', 'Kapalı'],
                yticklabels=['Tam Açık', 'Parçalı', 'Kapalı'],
                cbar_kws={'label': 'Sayı'},
                linewidths=2, linecolor='#334155',
                ax=ax)
    
    ax.set_title('Confusion Matrix', color='#e2e8f0', fontsize=16, fontweight='bold', pad=20)
    ax.set_xlabel('Tahmin Edilen', color='#94a3b8', fontsize=13, fontweight='bold')
    ax.set_ylabel('Gerçek Durum', color='#94a3b8', fontsize=13, fontweight='bold')
    ax.tick_params(colors='#64748b')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='#0f172a')
    print(f"✅ Confusion matrix kaydedildi: {save_path}")
    plt.show()
    
# ═══════════════════════════════════════════════════════════
# EĞİTİM
# ═══════════════════════════════════════════════════════════

def egitim_grafikleri_ciz(history, save_path='training_plots.png'):
    """Eğitim accuracy ve loss grafiklerini çizer"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    fig.patch.set_facecolor('#0f172a')
    
    epochs_range = range(1, len(history.history['accuracy']) + 1)
    
    # Accuracy
    ax1.set_facecolor('#1e293b')
    ax1.plot(epochs_range, [x*100 for x in history.history['accuracy']], 
             'o-', label='Training Accuracy', color='#10b981', linewidth=2, markersize=6)
    ax1.plot(epochs_range, [x*100 for x in history.history['val_accuracy']], 
             's-', label='Validation Accuracy', color='#f59e0b', linewidth=2, markersize=6)
    ax1.set_xlabel('Epoch', color='#94a3b8', fontsize=11)
    ax1.set_ylabel('Accuracy (%)', color='#94a3b8', fontsize=11)
    ax1.set_title('Model Accuracy', color='#e2e8f0', fontsize=14, fontweight='bold')
    ax1.legend(facecolor='#1e293b', labelcolor='#94a3b8', fontsize=10)
    ax1.grid(True, color='#334155', alpha=0.3)
    ax1.tick_params(colors='#64748b')
    
    # Loss
    ax2.set_facecolor('#1e293b')
    ax2.plot(epochs_range, history.history['loss'], 
             'o-', label='Training Loss', color='#10b981', linewidth=2, markersize=6)
    ax2.plot(epochs_range, history.history['val_loss'], 
             's-', label='Validation Loss', color='#f59e0b', linewidth=2, markersize=6)
    ax2.set_xlabel('Epoch', color='#94a3b8', fontsize=11)
    ax2.set_ylabel('Loss', color='#94a3b8', fontsize=11)
    ax2.set_title('Model Loss', color='#e2e8f0', fontsize=14, fontweight='bold')
    ax2.legend(facecolor='#1e293b', labelcolor='#94a3b8', fontsize=10)
    ax2.grid(True, color='#334155', alpha=0.3)
    ax2.tick_params(colors='#64748b')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='#0f172a')
    print(f"✅ Eğitim grafikleri kaydedildi: {save_path}")
    plt.show()


def modeli_egit(X, y):
    """CNN modelini eğitir"""
    
    print(f"\n{'='*60}")
    print(f"🎯 MODEL EĞİTİMİ BAŞLIYOR")
    print(f"{'='*60}\n")
    
    # Veriyi karıştır
    X, y = shuffle(X, y, random_state=42)
    
    # Train/validation split
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=VALIDATION_SPLIT, random_state=42, stratify=y
    )
    
    print(f"📊 Veri Bölümlemesi:")
    print(f"  Training: {len(X_train)} görüntü")
    print(f"  Validation: {len(X_val)} görüntü")
    
    # Sınıf dağılımı
    unique_train, counts_train = np.unique(y_train, return_counts=True)
    unique_val, counts_val = np.unique(y_val, return_counts=True)
    print(f"\n  Training dağılımı: 0:{counts_train[0]}, 1:{counts_train[1]}, 2:{counts_train[2]}")
    print(f"  Validation dağılımı: 0:{counts_val[0]}, 1:{counts_val[1]}, 2:{counts_val[2]}")
    
   # 🆕 CLASS WEIGHTS HESAPLA
    class_weights = compute_class_weight(
        'balanced',
        classes=np.unique(y_train),
        y=y_train
    )
    class_weight_dict = dict(enumerate(class_weights))
    
    #Manuel ayarlama (Parçalı ve Kapalı'ya daha fazla ağırlık)
    class_weight_dict = {
    0: 0.8,  # Tam Açık (azalt)
    1: 1.5,  # Parçalı (artır)
    2: 1.2   # Kapalı (artır)
    }
    
    print(f"\n⚖️ Class Weights (Dengesiz veri düzeltmesi):")
    print(f"  Tam Açık (0): {class_weights[0]:.3f}")
    print(f"  Parçalı (1): {class_weights[1]:.3f}")
    print(f"  Kapalı (2): {class_weights[2]:.3f}")
    
    # 🆕 DATA AUGMENTATION
    if USE_AUGMENTATION:
        print(f"\n🔄 Data Augmentation: AÇIK ({AUGMENTATION_STRENGTH} mode)")
        aug_config = get_augmentation_config(AUGMENTATION_STRENGTH)
        datagen = ImageDataGenerator(**aug_config)
        datagen.fit(X_train)
    else:
        print(f"\n🔄 Data Augmentation: KAPALI")
        datagen = None
    
    # Modeli oluştur
    print(f"\n🏗️ Model oluşturuluyor (İYİLEŞTİRİLMİŞ)...")
    print(f"  - L2 Regularization: {L2_WEIGHT}")
    print(f"  - Dropout (Conv): {DROPOUT_CONV}")
    print(f"  - Dropout (Dense): {DROPOUT_DENSE}")
    
    model = cnn_model_olustur(input_shape=(*IMG_SIZE, 3), num_classes=3)
    
    # Model özeti
    print(f"\n📋 Model Özeti:")
    model.summary()
    
    # Compile
    optimizer = keras.optimizers.Adam(learning_rate=LEARNING_RATE)
    model.compile(
        optimizer=optimizer,
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    # Callbacks
    callback_list = [
        callbacks.ModelCheckpoint(
            MODEL_SAVE_PATH,
            monitor='val_accuracy',
            save_best_only=True,
            verbose=1,
            mode='max'
        ),
        callbacks.EarlyStopping(
            monitor='val_loss',
            patience=25,
            restore_best_weights=True,
            verbose=1
        ),
        callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=7,
            min_lr=1e-7,
            verbose=1
        )
    ]
    
    # Eğitim
    print(f"\n{'='*60}")
    print(f"🚀 EĞİTİM BAŞLATILIYOR ({EPOCHS} epoch)")
    print(f"{'='*60}\n")
    
    log_dir = "logs/fit/" + datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=1)

    if USE_AUGMENTATION and datagen:
        # 🆕 Augmentation ile eğitim
        history = model.fit(
            datagen.flow(X_train, y_train, batch_size=BATCH_SIZE),
            validation_data=(X_val, y_val),
            epochs=EPOCHS,
            callbacks=callback_list,
            class_weight=class_weight_dict,  # 🆕
            verbose=1
        )
    else:
        # Normal eğitim
        history = model.fit(
            X_train,
            y_train,
            validation_data=(X_val, y_val),
            epochs=EPOCHS,
            batch_size=BATCH_SIZE,
            callbacks=callback_list,
            class_weight=class_weight_dict,  # 🆕
            verbose=1
        )
    
    print(f"\n{'='*60}")
    print(f"✅ EĞİTİM TAMAMLANDI!")
    print(f"{'='*60}")
    
    # Son değerlendirme
    print(f"\n📊 Final Değerlendirme:")
    train_loss, train_acc = model.evaluate(X_train, y_train, verbose=0)
    val_loss, val_acc = model.evaluate(X_val, y_val, verbose=0)
    
    print(f"  Training Accuracy: {train_acc*100:.2f}%")
    print(f"  Validation Accuracy: {val_acc*100:.2f}%")
    print(f"  Training Loss: {train_loss:.4f}")
    print(f"  Validation Loss: {val_loss:.4f}")
    
    
    # 🆕 CLASSIFICATION REPORT
    print(f"\n📋 Classification Report (Validation Set):")
    y_val_pred = np.argmax(model.predict(X_val, verbose=0), axis=1)
    report = classification_report(y_val, y_val_pred, 
                                   target_names=['Tam Açık', 'Parçalı', 'Kapalı'],
                                   digits=3)
    print(report)
    
    # 🆕 CONFUSION MATRIX
    print(f"\n📊 Confusion Matrix oluşturuluyor...")
    plot_confusion_matrix(y_val, y_val_pred, save_path='confusion_matrix.png')
    # History kaydet
    history_dict = {
        'accuracy': [float(x) for x in history.history['accuracy']],
        'val_accuracy': [float(x) for x in history.history['val_accuracy']],
        'loss': [float(x) for x in history.history['loss']],
        'val_loss': [float(x) for x in history.history['val_loss']],
        'epochs': len(history.history['accuracy']),
        'final_train_acc': float(train_acc),
        'final_val_acc': float(val_acc),
        'best_val_acc': float(max(history.history['val_accuracy'])),
        'img_size': IMG_SIZE,
        'batch_size': BATCH_SIZE,
        'learning_rate': LEARNING_RATE,
        'augmentation': USE_AUGMENTATION,  # 🆕
        'augmentation_strength': AUGMENTATION_STRENGTH if USE_AUGMENTATION else None,  # 🆕
        'l2_weight': L2_WEIGHT,  # 🆕
        'dropout_conv': DROPOUT_CONV,  # 🆕
        'dropout_dense': DROPOUT_DENSE,  # 🆕
        'class_weights': {str(k): float(v) for k, v in class_weight_dict.items()},  # 🆕
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    }
    
    with open(HISTORY_SAVE_PATH, 'w', encoding='utf-8') as f:
        json.dump(history_dict, f, indent=2, ensure_ascii=False)
    
    print(f"\n✅ Eğitim geçmişi kaydedildi: {HISTORY_SAVE_PATH}")
    
    # Grafikleri çiz
    print(f"\n📊 Eğitim grafikleri oluşturuluyor...")
    egitim_grafikleri_ciz(history)
    
    print(f"\n{'='*60}")
    print(f"🎉 TÜM İŞLEMLER TAMAMLANDI!")
    print(f"{'='*60}")
    print(f"✅ Model: {MODEL_SAVE_PATH}")
    print(f"✅ History: {HISTORY_SAVE_PATH}")
    print(f"✅ En iyi validation accuracy: {max(history.history['val_accuracy'])*100:.2f}%")
    print(f"{'='*60}\n")
    
    return model, history, X_val, y_val


# ═══════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("\n" + "="*60)
    print("🔭 TUG GÖKYÜZÜ CNN MODELİ EĞİTİMİ")
    print("Türkiye Ulusal Gözlemevleri")
    print("="*60)
    
    print(f"\n⚙️ Eğitim Ayarları:")
    print(f"  Veri Yolu: {DATA_PATH}")
    print(f"  Görüntü Boyutu: {IMG_SIZE}")
    print(f"  Batch Size: {BATCH_SIZE}")
    print(f"  Epochs: {EPOCHS}")
    print(f"  Learning Rate: {LEARNING_RATE}")
    print(f"  Validation Split: {VALIDATION_SPLIT}")
    print(f"  Sınıf Eşikleri: Tam Açık<{TAM_ACIK_ESIK}%, Parçalı<{PARCALI_ESIK}%")
    
    # Veri setini oluştur
    print(f"\n⏳ Veri seti hazırlanıyor...")
    X, y, dosya_listesi = veri_seti_olustur(
        DATA_PATH,
        max_per_class=1000,  # Her sınıftan max 1000 görüntü (hızlı eğitim için)
        verbose=True
    )
    
    # Modeli eğit
    model, history, X_val, y_val = modeli_egit(X, y)
    
   # 🆕 BATCH PREDICTION (İSTEĞE BAĞLI)
    print(f"\n{'='*60}")
    print(f"🔮 BATCH PREDICTION (Test Klasörü Analizi)")
    print(f"{'='*60}")
    
    test_klasor = input("\nTest klasörü yolu girin (Enter=atla): ").strip()
    
    if test_klasor and Path(test_klasor).exists():
        print(f"\n⏳ Test klasörü analizi yapılıyor: {test_klasor}")
        
        test_files = []
        for ext in UZANTILAR:
            test_files.extend(list(Path(test_klasor).rglob(f"*{ext}")))
        
        test_files = test_files[:100]  # İlk 100 dosya
        
        if test_files:
            predictions = []
            print(f"📂 {len(test_files)} görüntü bulundu\n")
            
            for i, img_path in enumerate(test_files, 1):
                img = gorsel_yukle_ve_hazirla(img_path, IMG_SIZE)
                if img is None:
                    continue
                
                img_batch = np.expand_dims(img, axis=0)
                pred = model.predict(img_batch, verbose=0)
                pred_class = int(np.argmax(pred))
                confidence = float(pred[0][pred_class])
                
                class_names = ['Tam Açık', 'Parçalı', 'Kapalı']
                
                predictions.append({
                    'dosya': img_path.name,
                    'tahmin': class_names[pred_class],
                    'guven': f"{confidence*100:.2f}%"
                })
                
                if i % 10 == 0 or i == len(test_files):
                    print(f"  [{i}/{len(test_files)}] {img_path.name}: {class_names[pred_class]} ({confidence*100:.1f}%)")
            
            if predictions:
                df = pd.DataFrame(predictions)
                csv_path = 'batch_predictions.csv'
                df.to_csv(csv_path, index=False, encoding='utf-8-sig')
                
                # Özet
                print(f"\n{'='*60}")
                print(f"📊 SONUÇLAR")
                print(f"{'='*60}")
                for i, name in enumerate(['Tam Açık', 'Parçalı', 'Kapalı']):
                    count = df[df['tahmin'] == name].shape[0]
                    print(f"  {['🟢', '🟡', '🔴'][i]} {name}: {count} görüntü ({count/len(df)*100:.1f}%)")
                
                print(f"\n✅ Sonuçlar kaydedildi: {csv_path}")
                print(f"{'='*60}\n")
        else:
            print("⚠️ Test klasöründe görüntü bulunamadı!")
    else:
        print("⏭️ Batch prediction atlandı")
    
    print("\n🎊 Program başarıyla tamamlandı!\n")