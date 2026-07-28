# 🎯 OmahTI Academy Challenge – Data Science & Artificial Intelligence (DSAI)

## Deskripsi Tantangan

Selamat datang di **OmahTI Academy Challenge** untuk Divisi **Data Science & Artificial Intelligence (DSAI)**.

Pada tantangan ini, peserta akan mengerjakan sebuah permasalahan *machine learning* yang nyata, yaitu **prediksi harga Airbnb di Singapura**. Seiring berkembangnya platform penyewaan jangka pendek seperti Airbnb, kemampuan untuk memperkirakan harga sewa secara akurat menjadi semakin penting bagi pemilik properti, wisatawan, maupun pembuat kebijakan.

Tugas utama peserta adalah membangun model *machine learning* yang mampu memprediksi **harga sewa per malam** dari sebuah listing Airbnb di Singapura.

Dataset yang disediakan bersifat **komprehensif**, karena menggabungkan dua jenis data, yaitu:

- **Data tabular**, seperti:
  - Karakteristik properti
  - Informasi host
  - Skor dan metrik ulasan
  - Fitur numerik maupun kategorikal lainnya
- **Data teks**, seperti:
  - Deskripsi listing
  - Ulasan atau komentar tamu

Pendekatan ini memungkinkan peserta untuk mengeksplorasi pemanfaatan baik **fitur terstruktur** maupun **fitur tekstual** dalam membangun model prediksi yang lebih akurat.

---

# 📌 Latar Belakang

Pertumbuhan platform penyewaan jangka pendek seperti Airbnb telah mengubah industri akomodasi di berbagai kota besar dunia, termasuk Singapura.

Harga sebuah listing dipengaruhi oleh banyak faktor, antara lain:

- Lokasi properti
- Jenis dan kapasitas properti
- Fasilitas yang tersedia
- Reputasi host
- Rating dan ulasan pelanggan
- Informasi lain yang terdapat pada deskripsi listing

Prediksi harga yang akurat memberikan manfaat bagi berbagai pihak, di antaranya:

- **Host**, untuk menentukan strategi harga yang kompetitif.
- **Wisatawan**, agar dapat membuat keputusan pemesanan yang lebih tepat.
- **Pembuat kebijakan**, untuk memahami dinamika pasar penyewaan jangka pendek.

Melalui tantangan ini, peserta diharapkan mampu mengembangkan solusi berbasis *machine learning* yang dapat memanfaatkan informasi dari berbagai jenis data secara optimal.

---

# 🎯 Tujuan

Membangun model prediksi yang mampu memperkirakan **harga sewa per malam** (*nightly price*) listing Airbnb di Singapura dengan tingkat kesalahan sekecil mungkin.

---

# 📊 Evaluasi

Performa model akan dievaluasi menggunakan metrik **Mean Absolute Percentage Error (MAPE)**.

MAPE mengukur rata-rata persentase kesalahan antara nilai prediksi dan nilai sebenarnya.

Semakin kecil nilai **MAPE**, maka semakin baik performa model yang dihasilkan.

## Rumus MAPE

\[
\text{MAPE} =
\frac{100\%}{n}
\sum_{i=1}^{n}
\left|
\frac{y_i-\hat{y}_i}{y_i}
\right|
\]

### Keterangan

| Simbol | Keterangan |
|--------|------------|
| \(n\) | Jumlah total data (observasi). |
| \(y_i\) | Harga sebenarnya (*actual price*) dari listing ke-*i*. |
| \(\hat{y}_i\) | Harga hasil prediksi (*predicted price*) dari listing ke-*i*. |

---

# 🚀 Target Peserta

Peserta diharapkan mampu:

- Melakukan eksplorasi data (*Exploratory Data Analysis / EDA*).
- Melakukan *preprocessing* data tabular maupun data teks.
- Melakukan *feature engineering* untuk meningkatkan kualitas fitur.
- Membangun model regresi menggunakan berbagai algoritma *machine learning*.
- Mengevaluasi model menggunakan metrik MAPE.
- Mengoptimalkan model agar memperoleh tingkat kesalahan prediksi serendah mungkin.

---

# 💡 Hasil Akhir

Output yang diharapkan dari model adalah sebuah nilai **harga sewa per malam** untuk setiap listing Airbnb pada data uji (*test set*).

Model terbaik adalah model yang menghasilkan **nilai MAPE paling rendah**, sehingga mampu memberikan prediksi harga yang paling mendekati nilai sebenarnya.