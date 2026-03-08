# FINANCE SAKU WITH SMART EXPENSE CATEGORIZER

Aplikasi keuangan pribadi berbasis Streamlit untuk mencatat pemasukan dan pengeluaran, lengkap dengan login, dashboard interaktif, auto-category transaksi, serta laporan dengan export CSV/PDF.

## Fitur Utama

- Login dan registrasi user (email + password hash)
- Penyimpanan data lokal menggunakan SQLite (`finance_saku.db`)
- Dashboard interaktif:
  - Total pemasukan, total pengeluaran, saldo, jumlah transaksi
  - Grafik tren bulanan (Plotly)
  - Pie chart komposisi kategori pengeluaran
- Manajemen transaksi:
  - Tambah transaksi pemasukan/pengeluaran
  - Auto-category untuk pengeluaran (model ML + fallback keyword)
  - Edit dan hapus transaksi
  - Filter tanggal, jenis transaksi, kategori, dan pencarian deskripsi
- Laporan:
  - Ringkasan bulanan
  - Grafik perbandingan bulanan
  - Export ke CSV dan PDF

## Teknologi

- Python + Streamlit
- SQLite (`sqlite3`)
- Pandas + NumPy
- Plotly
- ReportLab (PDF)
- Scikit-learn (model + vectorizer)

## Struktur Project

```text
smart-expense-categorizer/
|- app.py
|- model.pkl
|- vectorizer.pkl
|- requirements.txt
|- README.md
|- finance_saku.db          # otomatis dibuat saat aplikasi dijalankan
```

## Cara Menjalankan

1. (Opsional) Aktifkan virtual environment (Windows PowerShell):

```powershell
.\.venv\Scripts\Activate.ps1
```

2. Install dependency:

```bash
pip install -r requirements.txt
```

3. Jalankan aplikasi:

```powershell
python -m streamlit run app.py
```

4. Buka URL lokal yang ditampilkan Streamlit di browser.

## Alur Penggunaan

1. Registrasi akun baru.
2. Login menggunakan email dan password.
3. Tambah transaksi pada menu `Transaksi`.
4. Lihat ringkasan pada menu `Dashboard`.
5. Unduh laporan di menu `Laporan` (CSV/PDF).

## Catatan Auto Category

- Jika `model.pkl` dan `vectorizer.pkl` tersedia, kategori pengeluaran diprediksi oleh model ML.
- Jika model tidak tersedia atau gagal dimuat, aplikasi otomatis menggunakan fallback berbasis kata kunci.

## Catatan Penting

- Data tersimpan lokal pada file `finance_saku.db`.
- Tidak menggunakan API eksternal.
- Aplikasi berjalan sepenuhnya di mesin lokal.
