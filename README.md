# Case-Based Reasoning untuk Prediksi Putusan Hukum Narkotika

Proyek ini adalah implementasi sistem *Case-Based Reasoning* (CBR) untuk menganalisis dan memprediksi putusan hukum terkait kasus narkotika dari direktori putusan Mahkamah Agung RI, dengan fokus pada Pengadilan Negeri Surabaya.

Sistem ini melakukan beberapa tahapan:
1.  **Web Scraping**: Mengumpulkan data putusan dari situs Mahkamah Agung.
2.  **Case Representation**: Mengekstrak informasi relevan dan menyimpannya dalam format terstruktur (CSV/JSON).
3.  **Vector Representation**: Mengubah teks putusan menjadi representasi vektor menggunakan TF-IDF dan model IndoBERT.
4.  **Case Retrieval**: Menemukan kasus-kasus yang paling mirip dengan sebuah query (kasus baru) menggunakan *cosine similarity*.
5.  **Solution Reuse**: Memprediksi putusan (amar) untuk kasus baru berdasarkan kasus-kasus serupa yang ditemukan.
6.  **Evaluation**: Mengevaluasi performa model retrieval (TF-IDF vs. IndoBERT).

---

## ⚙️ Instalasi

Untuk menjalankan proyek ini di lingkungan lokal, ikuti langkah-langkah berikut.

1.  **Clone Repositori (Opsional)**
    Jika kode ini berada di dalam sebuah repositori Git, clone terlebih dahulu.
    ```bash
    git clone [URL-repositori-anda]
    cd [nama-direktori-proyek]
    ```

2.  **Buat Virtual Environment (Sangat Direkomendasikan)**
    Ini akan mengisolasi dependensi proyek Anda.
    ```bash
    python -m venv venv
    source venv/bin/activate  # Di Windows, gunakan: venv\Scripts\activate
    ```

3.  **Instal Dependensi**
    Instal semua library yang dibutuhkan menggunakan file `requirements.txt`.
    ```bash
    pip install -r requirements.txt
    ```

---

## 🚀 Cara Menjalankan Pipeline (End-to-End)

Seluruh pipeline diimplementasikan dalam Jupyter Notebook `PN_Modelling_2_SBY (1).ipynb`. Disarankan untuk menjalankannya di **Google Colab** untuk memanfaatkan GPU gratis (mempercepat pembuatan embedding BERT) dan menghindari masalah instalasi dependensi yang kompleks seperti PyTorch.

Jalankan sel-sel notebook secara berurutan dari atas ke bawah. Berikut adalah rincian setiap tahapannya:

### **Tahap 1 & 2: Pengumpulan dan Representasi Data**
*   **Tujuan**: Melakukan scraping data putusan, mengunduh file PDF, mengekstrak teks, dan menyimpannya dalam format CSV yang bersih.
*   **Konfigurasi**:
    *   Pastikan path Google Drive di sel kedua (`drive.mount`) sudah benar.
    *   Pada sel *main driver*, atur variabel `TARGET_URL` untuk menentukan query pencarian di situs Mahkamah Agung.
    *   Atur `max_cases` untuk membatasi jumlah putusan yang akan di-scrape.
*   **Output**:
    *   Direktori `/pdf/` berisi file-file PDF putusan.
    *   Direktori `/data/raw/` berisi file teks mentah dari setiap putusan.
    *   File `/data/processed/cases.csv` berisi data terstruktur dari semua putusan yang berhasil diproses.

### **Tahap 3: Representasi Vektor**
*   **Tujuan**: Mengubah data teks dari `cases.csv` menjadi vektor numerik menggunakan dua metode:
    1.  **TF-IDF**: Model berbasis frekuensi kata. Cepat tetapi tidak memahami konteks.
    2.  **IndoBERT**: Model Transformer yang memahami konteks dan semantik bahasa Indonesia. Lebih akurat tetapi membutuhkan waktu lebih lama untuk memproses.
*   **Output**:
    *   `/output/tfidf_vectorizer.pkl`: Model TF-IDF yang sudah di-train.
    *   `/output/tfidf_matrix.pkl`: Matriks TF-IDF dari seluruh korpus data.
    *   `/output/bert_embeddings.npy`: Matriks embedding dari IndoBERT.

### **Tahap 4: Prediksi (Solution Reuse)**
*   **Tujuan**: Mendefinisikan fungsi `predict_outcome` untuk memprediksi amar putusan sebuah kasus baru berdasarkan *voting* (mayoritas atau berbobot) dari kasus-kasus terdekat.
*   **Contoh Perintah**:
    Sel di tahap ini mendemonstrasikan cara menggunakan fungsi `predict_outcome` dengan beberapa contoh query manual.
    ```python
    # Contoh penggunaan fungsi prediksi
    predicted_solution, top_5_ids = predict_outcome(
        query="Seorang pria ditangkap karena menjadi kurir narkotika.",
        k=5,
        retrieval_method='bert',
        prediction_method='weighted'
    )
    print(f"Predicted Amar: {predicted_solution}")
    ```
*   **Output**:
    *   File `/data/results/predictions.csv` berisi hasil prediksi dari query-query demo.

### **Tahap 5: Evaluasi**
*   **Tujuan**: Mengukur dan membandingkan performa kedua model retrieval (TF-IDF dan IndoBERT) menggunakan metrik `Precision@k`, `Recall@k`, dan `F1-Score@k`.
*   **Konfigurasi**:
    *   File `/data/eval/queries.json` berisi query-query untuk pengujian beserta *ground truth* (ID kasus yang relevan secara manual).
*   **Output**:
    *   Tabel dan grafik perbandingan performa.
    *   File `/data/eval/retrieval_metrics.csv` berisi ringkasan metrik performa rata-rata.
    *   File `/data/eval/prediction_metrics.csv` berisi skor similaritas antara prediksi dan amar sebenarnya.