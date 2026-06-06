# Recruitly: CV & Job Description Matching using NLP

Proyek ini bertujuan untuk membangun model Natural Language Processing (NLP) guna melakukan pencocokan (*matching*) otomatis antara CV (Resume) pelamar dengan Deskripsi Pekerjaan (*Job Description*). Model memprediksi tingkat kecocokan dalam skala 1 hingga 5 (`match_score`) menggunakan pendekatan Supervised Learning dan Unsupervised Learning.

---

## 📂 Struktur Repositori

Repositori ini dibagi menjadi dua folder utama berdasarkan pendekatan metodologi NLP yang digunakan:

1. **`NLP_Supervised/` (Aktif)**
   * Berisi eksperimen, eksplorasi data, dan pemodelan regresi terbimbing (*supervised*) untuk memprediksi `match_score` berdasarkan data latih yang memiliki label kecocokan (1-5).
2. **`NLP_Unsupervised/` (Segera Datang / Coming Soon)**
   * Akan berisi pendekatan tidak terbimbing (*unsupervised*) seperti clustering, pencarian kemiripan semantik murni (*semantic search*), atau pemodelan topik (*topic modeling*) tanpa bergantung pada label skor kecocokan historis.

---

## 🛠️ Detail Folder: `NLP_Supervised`

Di dalam folder ini, terdapat beberapa notebook utama yang mendokumentasikan alur pengembangan model:

### 1. Eksplorasi Data (`NLP_Supervised/recruitly_dataset_exploration.ipynb`)
* Analisis awal dataset [resume_job_matching_dataset.csv](file:///d:/Hasil_Coding/NLP_Supervised/NLP_Supervised/resume_job_matching_dataset.csv).
* Visualisasi distribusi target `match_score` dan pemahaman awal karakteristik teks pada CV serta *Job Description*.

### 2. Pendekatan Regresi Tradisional dengan TF-IDF (`NLP_Supervised/recruitly_resume_screening_regression.ipynb`)
* **Ekstraksi Fitur**: Pembobotan TF-IDF dan kalkulasi *Cosine Similarity* berbasis kata kunci (*keyword matching*).
* **Fitur Regresi**: Menggabungkan nilai *Cosine Similarity* dengan nilai perbedaan absolut (*absolute difference*) dari vektor TF-IDF.
* **Model yang Dievaluasi**: Linear Regression, Random Forest, XGBoost, dan LightGBM.

### 3. Pendekatan Regresi Semantik dengan SBERT (`NLP_Supervised/recruitly_resume_screening_sbert.ipynb`)
* **Ekstraksi Fitur**: Menggunakan pre-trained model **Sentence-BERT (`all-MiniLM-L6-v2`)** untuk menghasilkan embedding 384 dimensi yang menangkap makna kontekstual kalimat secara utuh (bukan hanya pencocokan kata kunci).
* **Model & Regularisasi**: Menggunakan Linear Regression, **Ridge Regression**, **Lasso Regression**, Random Forest, XGBoost, dan LightGBM.
* **Tuning**: Optimasi parameter `alpha` pada Ridge Regression menggunakan `GridSearchCV` 5-fold cross-validation.

---

## 📊 Metrik Evaluasi

Untuk mengukur performa model regresi dalam tugas *screening* dan *ranking* CV, metrik berikut digunakan:
* **MAE (Mean Absolute Error) & RMSE (Root Mean Squared Error)**: Mengukur rata-rata kesalahan prediksi numerik absolut terhadap skor 1-5 asli.
* **R-squared ($R^2$)**: Mengukur seberapa besar varians target yang mampu dijelaskan oleh fitur.
* **Korelasi Spearman (Spearman's Rank Correlation)**: Mengukur kualitas korelasi arah peringkat (*ranking direction*) antar kandidat. Sangat krusial agar recruiter mendapatkan urutan CV terbaik secara logis.
* **NDCG (Normalized Discounted Cumulative Gain)**: Metrik utama sistem rekomendasi/search engine untuk mengevaluasi ketepatan penempatan CV dengan kecocokan tinggi di urutan teratas (*top rankings*).

---

## ⚙️ Cara Memulai & Setup Lokal

Proyek ini dikonfigurasi menggunakan *Virtual Environment* lokal di Drive D untuk menghindari penuhnya penyimpanan Drive C Anda.

### 1. Kloning Repositori
```bash
git clone https://github.com/FelliAs/recruitly-test.git
cd recruitly-test
```

### 2. Setup Virtual Environment (`venv`)
Buat dan aktifkan `venv` di dalam root direktori:
```powershell
# Membuat venv
python -m venv venv

# Mengaktifkan venv (Windows PowerShell)
.\venv\Scripts\Activate.ps1
```

### 3. Instalasi Dependensi
Instal PyTorch dengan dukungan akselerasi GPU (CUDA) dan library pendukung lainnya:
```powershell
# Instal PyTorch GPU (CUDA 12.1)
pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cu121

# Instal library pendukung
pip install --no-cache-dir sentence-transformers pandas numpy scikit-learn lightgbm xgboost matplotlib seaborn ipykernel
```

### 4. Menghindari Kernel Crash & Cache C-Drive
* **Konfigurasi Cache**: Folder cache Hugging Face secara otomatis dialihkan ke Drive D melalui environment variable `HF_HOME` di dalam notebook agar tidak memakan memori Drive C.
* **Konfigurasi OpenMP**: Variabel lingkungan `KMP_DUPLICATE_LIB_OK = "TRUE"` diterapkan di awal notebook untuk menghindari *kernel crash* akibat konflik inisialisasi library OpenMP di Windows.
