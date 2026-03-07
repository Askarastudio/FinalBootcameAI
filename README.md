# SMART EXPENSE CATEGORIZER

Aplikasi web berbasis Streamlit untuk mengklasifikasikan teks transaksi menjadi kategori pengeluaran secara otomatis menggunakan model Machine Learning (`model.pkl`) dan vectorizer TF-IDF (`vectorizer.pkl`).

## Fitur Utama

- Prediksi kategori transaksi secara otomatis
- Preprocessing teks dengan regex (`clean_text`)
- Confidence score (jika model mendukung `predict_proba`)
- Dashboard UI modern (custom CSS, card, metric, sidebar)
- Contoh transaksi cepat
- Riwayat prediksi dalam sesi menggunakan `st.session_state`

## Struktur Project

```text
smart-expense-categorizer/
|- app.py
|- model.pkl
|- vectorizer.pkl
|- requirements.txt
|- README.md
```

## Cara Menjalankan

1. (Opsional) Aktivasi virtual environment di Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

2. Install dependency:

```bash
pip install -r requirements.txt
```

3. Jalankan aplikasi (cara yang paling aman):

```powershell
python -m streamlit run app.py
```

Alternatif jika command global tersedia:

```bash
streamlit run app.py
```

4. Buka browser pada URL lokal yang ditampilkan Streamlit.

## Catatan

- Tidak menggunakan database.
- Tidak menggunakan API eksternal.
- Semua berjalan lokal di mesin Anda.
- Default dependency memakai `scikit-learn==1.6.1` agar stabil di environment Python terbaru (termasuk Python 3.14).
- Jika model dibuat di versi scikit-learn yang lebih lama (mis. 1.6.1), aplikasi tetap bisa berjalan untuk demo.
