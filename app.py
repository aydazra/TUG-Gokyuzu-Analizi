import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# Sayfa ayarları
st.set_page_config(page_title="TUG Gökyüzü Sınıflandırma", layout="centered")

st.title("🔭 TUG Gökyüzü Bulut Sınıflandırma")
st.write("Bir gökyüzü görüntüsü yükleyin ve model tahmin yapsın.")

# Modeli yükle
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model("tug_gokyuzu_cnn_model.h5")
    return model

model = load_model()

# Sınıf isimleri (senin sıralamana göre düzenlenebilir)
class_names = ["Tam Açık", "Parçalı Bulutlu", "Kapalı"]

uploaded_file = st.file_uploader("📤 Görüntü yükleyin", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Yüklenen Görüntü", use_column_width=True)

    # Model için hazırla
    img = image.resize((128, 128))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array)
    predicted_class = np.argmax(prediction)
    confidence = np.max(prediction)

    st.success(f"📊 Tahmin: {class_names[predicted_class]}")
    st.info(f"🎯 Güven Oranı: %{confidence*100:.2f}")
