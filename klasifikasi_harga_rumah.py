import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.preprocessing import LabelEncoder

# Membaca dataset dengan delimiter ;
data = pd.read_csv("HARGA RUMAH JAKSEL.csv", delimiter=';')

# Konversi kolom HARGA ke tipe data numerik
data['HARGA'] = pd.to_numeric(data['HARGA'], errors='coerce')

# Hapus baris dengan nilai NaN
data = data.dropna(subset=['HARGA'])

# Menambahkan kolom kategori berdasarkan harga (contoh: harga di atas 10 miliar dianggap mahal)
batas_harga = 10000000000  # 10 miliar
data['KATEGORI'] = data['HARGA'].apply(lambda x: 'Mahal' if x > batas_harga else 'Murah')

# Encoding label KATEGORI menjadi numerik
label_encoder = LabelEncoder()
data['KATEGORI_ENCODED'] = label_encoder.fit_transform(data['KATEGORI'])

# Memisahkan fitur (X) dan label (y) untuk klasifikasi
X_class = data[['LUASTANAH', 'LUASBANGUNAN', 'JUMLAHKAMARTIDUR', 'JUMLAHKAMARMANDI']]
y_class = data['KATEGORI_ENCODED']

# Membagi dataset untuk klasifikasi
X_train_class, X_test_class, y_train_class, y_test_class = train_test_split(X_class, y_class, test_size=0.2, random_state=42)

# Melatih model Logistic Regression untuk klasifikasi
model_class = LogisticRegression()
model_class.fit(X_train_class, y_train_class)

# Memisahkan fitur (X) dan label (y) untuk regresi
X_reg = data[['LUASTANAH', 'LUASBANGUNAN', 'JUMLAHKAMARTIDUR', 'JUMLAHKAMARMANDI']]
y_reg = data['HARGA']

# Membagi dataset untuk regresi
X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(X_reg, y_reg, test_size=0.2, random_state=42)

# Melatih model Linear Regression untuk prediksi harga
model_reg = LinearRegression()
model_reg.fit(X_train_reg, y_train_reg)

# Aplikasi Streamlit
st.title("Klasifikasi dan Prediksi Harga Rumah")

# Input dari pengguna
st.subheader("Masukkan Detail Properti")
luas_tanah = st.number_input("Luas Tanah (m²)", min_value=0, value=500)
luas_bangunan = st.number_input("Luas Bangunan (m²)", min_value=0, value=300)
jumlah_kamar_tidur = st.number_input("Jumlah Kamar Tidur", min_value=1, value=3)
jumlah_kamar_mandi = st.number_input("Jumlah Kamar Mandi", min_value=1, value=2)

# Tombol untuk melakukan prediksi
if st.button("Prediksi"):
    # Membuat DataFrame dari input pengguna
    input_data = pd.DataFrame([[luas_tanah, luas_bangunan, jumlah_kamar_tidur, jumlah_kamar_mandi]],
                              columns=['LUASTANAH', 'LUASBANGUNAN', 'JUMLAHKAMARTIDUR', 'JUMLAHKAMARMANDI'])
    
    # Melakukan prediksi kategori harga
    prediksi_kategori = model_class.predict(input_data)[0]
    kategori = label_encoder.inverse_transform([prediksi_kategori])[0]
    
    # Melakukan prediksi harga
    prediksi_harga = model_reg.predict(input_data)[0]
    
    # Menampilkan hasil prediksi
    st.subheader("Hasil Prediksi")
    st.write(f"Harga Rumah ini termasuk kategori **{kategori}**.")
    st.write(f"Prediksi harga : **Rp {prediksi_harga:,.0f}**.")
