# -- coding: utf-8 --
"""
📱 Program Prediksi KATEGORI Kecanduan HP (Versi Klasifikasi)
Menggunakan Decision Tree dan Random Forest
Dengan Input Interaktif + Visualisasi Lengkap
"""

# ===== IMPORT LIBRARY =====
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report, accuracy_score
from sklearn.preprocessing import StandardScaler

import joblib     # <<<<<< WAJIB ADA DI SINI

warnings.filterwarnings("ignore")

# ===== 1. LOAD DATA =====
print("=" * 80)
print("📱 PROGRAM PREDIKSI KATEGORI KECANDUAN HP (File: teen_phone_addiction...)")
print("=" * 80)

try:
    dataset = pd.read_csv("teen_phone_addiction_dataset.csv", sep=";")
except FileNotFoundError:
    print("❌ ERROR: File 'teen_phone_addiction_dataset.csv' tidak ditemukan.")
    exit()

print("\n🧹 Mengubah data target (skor) menjadi kategori...")

# ===== 2. PILIH FITUR =====
features = [
    "Age",
    "Daily_Usage_Hours",
    "Phone_Checks_Per_Day",
    "Time_on_Social_Media",
    "Time_on_Gaming",
    "Sleep_Hours",
    "Exercise_Hours",
]

missing_features = [f for f in features if f not in dataset.columns]
if missing_features:
    print(f"❌ ERROR: Fitur berikut tidak ditemukan di CSV: {missing_features}")
    exit()

X = dataset[features]
y_numeric = dataset["Addiction_Level"]

# kategori
bins = [0, 5.0, 8.0, 10.0]
labels = ["Rendah", "Sedang", "Tinggi"]
y = pd.cut(y_numeric, bins=bins, labels=labels, right=True, include_lowest=True)

print("Distribusi Kategori:")
print(y.value_counts())

# ===== 3. SPLIT DATA =====
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print(f"\nTraining set: {len(X_train)} sampel")
print(f"Testing set : {len(X_test)} sampel")

# ===== 4. TRAIN DECISION TREE =====
print("\n🌳 Melatih Decision Tree...")

dt = DecisionTreeClassifier(
    max_depth=7,
    min_samples_leaf=8,
    min_samples_split=10,
    class_weight="balanced",
    random_state=42,
)
dt.fit(X_train_scaled, y_train)

dt_train_acc = dt.score(X_train_scaled, y_train)
dt_test_acc = dt.score(X_test_scaled, y_test)

print(f"Akurasi Train : {dt_train_acc:.3f}")
print(f"Akurasi Test  : {dt_test_acc:.3f}")

cv = KFold(n_splits=5, shuffle=True, random_state=42)
dt_cv_score = cross_val_score(dt, X_train_scaled, y_train, cv=cv).mean()
print(f"CV Accuracy   : {dt_cv_score:.3f}")

# ===== 5. TRAIN RANDOM FOREST =====
print("\n🌲 Melatih Random Forest...")

rf = RandomForestClassifier(
    n_estimators=200,
    max_depth=10,
    min_samples_leaf=10,
    min_samples_split=10,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1,
)
rf.fit(X_train_scaled, y_train)

rf_train_acc = rf.score(X_train_scaled, y_train)
rf_test_acc = rf.score(X_test_scaled, y_test)

print(f"Akurasi Train : {rf_train_acc:.3f}")
print(f"Akurasi Test  : {rf_test_acc:.3f}")

rf_cv_score = cross_val_score(rf, X_train_scaled, y_train, cv=cv).mean()
print(f"CV Accuracy   : {rf_cv_score:.3f}")

# ===== 6. VISUALISASI =====
print("\n📊 Membuat visualisasi...")

# Confusion Matrix
cm_dt = confusion_matrix(y_test, dt.predict(X_test_scaled), labels=labels)
cm_rf = confusion_matrix(y_test, rf.predict(X_test_scaled), labels=labels)

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
fig.suptitle("Confusion Matrix")

ConfusionMatrixDisplay(cm_dt, display_labels=labels).plot(cmap="Blues", ax=axes[0])
axes[0].set_title("Decision Tree")

ConfusionMatrixDisplay(cm_rf, display_labels=labels).plot(cmap="Greens", ax=axes[1])
axes[1].set_title("Random Forest")

plt.tight_layout()
plt.savefig("confusion_matrix.png")
plt.show()

# Feature Importance RF
fi_rf = pd.DataFrame({
    "Feature": features,
    "Importance": rf.feature_importances_
}).sort_values("Importance")

plt.figure(figsize=(8, 5))
plt.barh(fi_rf["Feature"], fi_rf["Importance"])
plt.title("Feature Importance - Random Forest")
plt.savefig("feature_importance_clf.png")
plt.show()

# Decision Tree plot
plt.figure(figsize=(22, 10))
plot_tree(
    dt,
    feature_names=features,
    class_names=labels,
    filled=True,
    rounded=True,
    fontsize=9,
)
plt.title("Decision Tree Visualization")
plt.savefig("decision_tree_clf.png")
plt.show()

# ===== 7. SIMPAN MODEL =====
joblib.dump(dt, "decision_tree_model.pkl")
joblib.dump(rf, "random_forest_model.pkl")
joblib.dump(scaler, "scaler.pkl")

print("\n💾 Model berhasil disimpan:")
print("- decision_tree_model.pkl")
print("- random_forest_model.pkl")
print("- scaler.pkl")

# ===== 8. INPUT INTERAKTIF =====
def get_user_input():
    print("\n📝 Masukkan Data Anda")
    inputs = {}
    questions = [
        ("Age", "Usia (tahun)", int),
        ("Daily_Usage_Hours", "Waktu layar per hari (jam)", float),
        ("Phone_Checks_Per_Day", "Berapa kali membuka HP", int),
        ("Time_on_Social_Media", "Waktu media sosial (jam)", float),
        ("Time_on_Gaming", "Waktu bermain game (jam)", float),
        ("Sleep_Hours", "Durasi tidur (jam)", float),
        ("Exercise_Hours", "Durasi aktivitas fisik (jam)", float),
    ]
    for key, prompt, dtype in questions:
        while True:
            try:
                val = dtype(input(f"{prompt}: "))
                if val >= 0:
                    inputs[key] = val
                    break
                else:
                    print("⚠ Masukkan angka positif.")
            except:
                print("⚠ Input tidak valid.")
    return inputs

def predict_user(data):
    arr = np.array([[data[f] for f in features]])
    arr_scaled = scaler.transform(arr)

    dt_pred = dt.predict(arr_scaled)[0]
    dt_proba = dt.predict_proba(arr_scaled)[0]
    rf_pred = rf.predict(arr_scaled)[0]
    rf_proba = rf.predict_proba(arr_scaled)[0]

    print("\n=== HASIL PREDIKSI ===")
    print("\n🌳 Decision Tree:", dt_pred)
    print("\n🌲 Random Forest:", rf_pred)

# ===== 9. MENU UTAMA =====
print("\n📋 MODEL SIAP DIGUNAKAN")
print(f"Akurasi Decision Tree : {dt_test_acc:.3f}")
print(f"Akurasi Random Forest : {rf_test_acc:.3f}")

while True:
    tanya = input("\nIngin prediksi? (y/n): ").lower()
    if tanya == "y":
        data = get_user_input()
        predict_user(data)
    else:
        print("\nTerima kasih! Gunakan HP dengan bijak 📱✨")
        break
