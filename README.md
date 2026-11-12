# Prediksi Niat Beli Pengunjung Toko Online dengan Random Forest (TA-06)

Eksplorasi Random Forest (pengaruh `n_estimators` dan `max_depth`, feature importance) pada data sesi kunjungan toko online, lengkap dengan aplikasi web Flask yang menampilkan prediksi **dan suara tiap pohon**.

Dataset: [Online Shoppers Purchasing Intention (Kaggle)](https://www.kaggle.com/datasets/henrysue/online-shoppers-intention). File yang sama juga tersedia di [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/468/online+shoppers+purchasing+intention+dataset) (`online_shoppers_intention.csv`).

## Struktur proyek

```
.
├── TA06_Random_Forest_Shoppers.ipynb   # preprocessing, eksperimen A & B, feature importance, simpan model
├── app.py                              # aplikasi Flask
├── templates/index.html                # halaman form + hasil (CSS ditanam di file ini)
├── data/online_shoppers_intention.csv  # letakkan dataset di sini
├── model/shoppers_random_forest.joblib # dihasilkan oleh notebook
├── requirements.txt                    # dihasilkan/ditimpa oleh notebook
└── Procfile                            # untuk Render / Railway
```

## 1. Latih model

1. Unduh CSV dari Kaggle, simpan ke folder `data/`.
2. Jalankan seluruh sel `TA06_Random_Forest_Shoppers.ipynb`.
3. Sel terakhir menghasilkan `model/shoppers_random_forest.joblib` dan `requirements.txt` dengan versi library yang tepat.
4. Jika memakai Colab, unduh kedua file itu ke folder proyek.

## 2. Jalankan lokal

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python app.py
```

Buka http://127.0.0.1:5000

## 3. Deploy

**Vercel:** Add New → Project → impor repositori ini. Vercel mendeteksi Flask dari `app.py`. Pastikan file model ikut ter-commit. CSS sengaja ditanam di `index.html` karena folder `static/` Flask tidak dilayani di Vercel.

**Render:** New → Web Service. Build Command `pip install -r requirements.txt`, Start Command `gunicorn app:app`.

**Railway:** Deploy from GitHub repo. `Procfile` dibaca otomatis.

## Catatan

- Versi `scikit-learn` saat deployment harus sama dengan saat training.
- Ukuran file model dicetak oleh notebook. Batas GitHub 100 MB per file; model dikompres (`compress=3`).
- Endpoint `/health` untuk memeriksa aplikasi berjalan.
# prediksi-niat-beli-pengunjung-online-shop
