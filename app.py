"""Aplikasi Flask: prediksi niat beli pengunjung toko online dengan Random Forest (TA-06)."""
from pathlib import Path

import joblib
import pandas as pd
from flask import Flask, redirect, render_template, request, url_for

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "model" / "shoppers_random_forest.joblib"

app = Flask(__name__)

if not MODEL_PATH.exists():
    raise FileNotFoundError(
        f"Model tidak ditemukan di {MODEL_PATH}. Jalankan notebook TA06 sampai bagian 9 lebih dulu."
    )

model = joblib.load(MODEL_PATH)
prep = model.named_steps["prep"]
clf = model.named_steps["clf"]

NUM_COLS = list(prep.transformers_[0][2])
CAT_COLS = list(prep.transformers_[1][2])
CATEGORIES = dict(zip(CAT_COLS, prep.named_transformers_["cat"].categories_))
FEATURE_NAMES = list(prep.get_feature_names_out())

# ---------------- Konfigurasi form ----------------
# key: (label, petunjuk, min, max, default, apakah persen)
NUMERIC_FIELDS = {
    "Administrative": ("Halaman akun/administrasi dibuka", "jumlah halaman", 0, 100, "2", False),
    "Administrative_Duration": ("Durasi di halaman akun", "detik", 0, 100000, "80", False),
    "Informational": ("Halaman informasi dibuka", "jumlah halaman (bantuan, kontak, dll.)", 0, 100, "0", False),
    "Informational_Duration": ("Durasi di halaman informasi", "detik", 0, 100000, "0", False),
    "ProductRelated": ("Halaman produk dibuka", "jumlah halaman", 0, 2000, "20", False),
    "ProductRelated_Duration": ("Durasi di halaman produk", "detik", 0, 200000, "800", False),
    "BounceRates": ("Bounce rate", "% sesi yang langsung keluar", 0, 100, "2", True),
    "ExitRates": ("Exit rate", "% keluar dari halaman yang dilihat", 0, 100, "4", True),
    "PageValues": ("Page value", "nilai rata-rata halaman sebelum transaksi", 0, 1000, "0", False),
}
SPECIAL_DAY_OPTIONS = [("0", "Tidak dekat hari spesial"), ("0.2", "0.2"), ("0.4", "0.4"),
                       ("0.6", "0.6"), ("0.8", "0.8"), ("1.0", "1.0 (tepat di hari spesial)")]
FIELD_GROUPS = [
    ("Aktivitas halaman", ["Administrative", "Administrative_Duration", "Informational",
                           "Informational_Duration", "ProductRelated", "ProductRelated_Duration"]),
    ("Kualitas kunjungan", ["BounceRates", "ExitRates", "PageValues", "SpecialDay"]),
    ("Konteks kunjungan", ["Month", "VisitorType", "Weekend"]),
]
LABELS = {k: v[0] for k, v in NUMERIC_FIELDS.items()}
LABELS.update({"SpecialDay": "Kedekatan dengan hari spesial", "Month": "Bulan kunjungan",
               "VisitorType": "Tipe pengunjung", "Weekend": "Kunjungan di akhir pekan"})

MONTH_ORDER = ["Jan", "Feb", "Mar", "Apr", "May", "June", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
MONTH_LABELS = {"Jan": "Januari", "Feb": "Februari", "Mar": "Maret", "Apr": "April", "May": "Mei",
                "June": "Juni", "Jul": "Juli", "Aug": "Agustus", "Sep": "September",
                "Oct": "Oktober", "Nov": "November", "Dec": "Desember"}
VISITOR_LABELS = {"Returning_Visitor": "Pengunjung lama (returning)", "New_Visitor": "Pengunjung baru",
                  "Other": "Lainnya"}


def month_options():
    present = list(CATEGORIES["Month"])
    ordered = [m for m in MONTH_ORDER if m in present] + [m for m in present if m not in MONTH_ORDER]
    return [(m, MONTH_LABELS.get(m, m)) for m in ordered]


def visitor_options():
    return [(v, VISITOR_LABELS.get(v, v)) for v in CATEGORIES["VisitorType"]]


def select_options(field):
    if field == "Month":
        return month_options()
    if field == "VisitorType":
        return visitor_options()
    if field == "Weekend":
        return [("0", "Tidak (hari kerja)"), ("1", "Ya (akhir pekan)")]
    if field == "SpecialDay":
        return SPECIAL_DAY_OPTIONS
    return []


def default_values():
    values = {k: v[4] for k, v in NUMERIC_FIELDS.items()}
    values["SpecialDay"] = "0"
    values["Weekend"] = "0"
    values["Month"] = "Nov" if "Nov" in CATEGORIES["Month"] else month_options()[0][0]
    values["VisitorType"] = ("Returning_Visitor" if "Returning_Visitor" in CATEGORIES["VisitorType"]
                             else visitor_options()[0][0])
    return values


# ---------------- Pengetahuan global model: feature importance ----------------
def original_name(name):
    for c in CAT_COLS:
        if name.startswith(f"cat__{c}_"):
            return c
    return name.split("__", 1)[1]


def global_importance(top=6):
    agg = {}
    for name, imp in zip(FEATURE_NAMES, clf.feature_importances_):
        key = original_name(name)
        agg[key] = agg.get(key, 0.0) + float(imp)
    ranked = sorted(agg.items(), key=lambda kv: kv[1], reverse=True)[:top]
    biggest = ranked[0][1] or 1.0
    return [{"label": LABELS.get(k, k), "value": round(v * 100, 1), "width": round(v / biggest * 100)}
            for k, v in ranked]


IMPORTANCE = global_importance()
N_TREES = len(clf.estimators_)


# ---------------- Validasi & prediksi ----------------
def parse_form(form):
    errors, row = [], {}
    for key, (label, _hint, lo, hi, _default, is_pct) in NUMERIC_FIELDS.items():
        raw = (form.get(key) or "").strip().replace(",", ".")
        try:
            val = float(raw)
        except ValueError:
            errors.append(f"{label} harus berupa angka.")
            continue
        if not lo <= val <= hi:
            errors.append(f"{label} harus antara {lo} dan {hi}.")
            continue
        row[key] = val / 100 if is_pct else val

    for key in ["SpecialDay", "Weekend"]:
        raw = form.get(key)
        if raw not in {v for v, _ in select_options(key)}:
            errors.append(f"Pilihan {LABELS[key]} tidak valid.")
        else:
            row[key] = float(raw)
    for key in CAT_COLS:
        raw = form.get(key)
        if raw not in CATEGORIES[key]:
            errors.append(f"Pilihan {LABELS[key]} tidak valid.")
        else:
            row[key] = raw
    return row, errors


def predict(row):
    X = pd.DataFrame([row], columns=NUM_COLS + CAT_COLS)
    proba = float(model.predict_proba(X)[0, list(clf.classes_).index(1)])
    pred = int(model.predict(X)[0])

    # Suara tiap pohon (ilustrasi cara kerja ensemble)
    Xt = prep.transform(X)
    votes = [int(clf.classes_[int(t.predict(Xt)[0])]) for t in clf.estimators_]
    yes = sum(votes)
    return {
        "buy": pred == 1,
        "label": "Berpotensi membeli" if pred == 1 else "Cenderung tidak membeli",
        "pct": round(proba * 100, 1),
        "yes": yes,
        "no": len(votes) - yes,
        "total": len(votes),
        "cells": [1] * yes + [0] * (len(votes) - yes),
    }


def render(values, result=None, errors=None):
    return render_template(
        "index.html", groups=FIELD_GROUPS, labels=LABELS, numeric=NUMERIC_FIELDS,
        select_options=select_options, values=values, result=result, errors=errors or [],
        importance=IMPORTANCE, n_trees=N_TREES,
    )


@app.get("/")
def index():
    return render(default_values())


@app.post("/predict")
def predict_view():
    values = default_values()
    values.update({k: v for k, v in request.form.items() if k in values})
    row, errors = parse_form(request.form)
    if errors:
        return render(values, errors=errors), 400
    return render(values, result=predict(row))


@app.get("/predict")
def predict_redirect():
    return redirect(url_for("index"))


@app.get("/health")
def health():
    return {"status": "ok", "trees": N_TREES}


if __name__ == "__main__":
    app.run(debug=True)
