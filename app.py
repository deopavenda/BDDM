
import streamlit as st
import joblib

# Load model
model = joblib.load('sentiment_model.pkl')

# Judul
st.title("🧠 Analisis Sentimen untuk Prediksi Tren Pasar")
st.write("Masukkan teks dari sosial media dan lihat prediksi arah pasar.")

# Input
user_input = st.text_area("📝 Masukkan teks")

if st.button("🔍 Prediksi"):
    if user_input.strip():
        pred = model.predict([user_input])[0]
        label = "📈 POSITIF (Bullish)" if pred == 1 else "📉 NEGATIF (Bearish)"
        st.success(f"Hasil Prediksi: {label}")
    else:
        st.warning("Teks tidak boleh kosong.")
