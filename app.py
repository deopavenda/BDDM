import streamlit as st
import joblib

# Load model dan vectorizer
model = joblib.load("random_forest_model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

# Tampilan web
st.title("📊 Analisis Sentimen Media Sosial untuk Prediksi Tren Pasar")
st.write("Model ini menggunakan Random Forest untuk memprediksi sentimen dari judul postingan sosial media (Reddit).")

# Input teks
user_input = st.text_area("Masukkan teks (judul posting Reddit atau berita saham):", "")

if st.button("Prediksi Sentimen"):
    if user_input.strip() == "":
        st.warning("Silakan masukkan teks terlebih dahulu.")
    else:
        text_vector = vectorizer.transform([user_input])
        prediction = model.predict(text_vector)[0]
        st.success(f"Prediksi Sentimen: **{prediction.upper()}**")
