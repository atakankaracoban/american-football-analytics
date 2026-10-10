import os
import matplotlib.pyplot as plt
import nfl_data_py as nfl
import numpy as np
import pandas as pd
import shap
from sklearn.metrics import (
    brier_score_loss,
    classification_report,
    confusion_matrix,
    roc_auc_score,
)
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold, train_test_split
from xgboost import XGBClassifier

from feature_prep import prepare_pre_snap_dataset

# Çıktı klasörünü doğrula
os.makedirs("output", exist_ok=True)


# ==========================================
# 1. VERİ YÜKLEME VE BÖLME
# ==========================================
print("1/5: 2023 Play-by-Play verisi çekiliyor ve değişkenler işleniyor...")
cols = [
    "game_id",
    "play_id",
    "play_type",
    "down",
    "ydstogo",
    "yardline_100",
    "qtr",
    "half_seconds_remaining",
    "score_differential",
    "shotgun",
    "no_huddle",
    "posteam",
    "defteam",
]
raw_pbp = nfl.import_pbp_data(
    [2023], columns=cols, downcast=True, include_participation=False
)
X, y, meta = prepare_pre_snap_dataset(raw_pbp)

# Train-Test Ayrımı (%80 Eğitim, %20 Test)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)
print(f"Eğitim Seti: {X_train.shape[0]} oyun | Test Seti: {X_test.shape[0]} oyun")


# ==========================================
# 2. HİPERPARAMETRE OPTİMİZASYONU
# ==========================================
print("\n2/5: Hiperparametre optimizasyonu yürütülüyor (RandomizedSearchCV)...")

param_dist = {
    "n_estimators": [100, 150, 200],
    "max_depth": [3, 4, 5, 6],
    "learning_rate": [0.03, 0.05, 0.1],
    "subsample": [0.8, 0.9, 1.0],
    "colsample_bytree": [0.7, 0.8, 0.9],
    "min_child_weight": [1, 3, 5],
}

base_xgb = XGBClassifier(
    objective="binary:logistic",
    eval_metric="logloss",
    random_state=42,
    tree_method="hist",
)

cv_strategy = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

search = RandomizedSearchCV(
    estimator=base_xgb,
    param_distributions=param_dist,
    n_iter=10,
    scoring="roc_auc",
    cv=cv_strategy,
    verbose=1,
    random_state=42,
    n_jobs=-1,
)
search.fit(X_train, y_train)

best_model = search.best_estimator_
print(f"En İyi Parametreler: {search.best_params_}")


# ==========================================
# 3. TEST DEĞERLENDİRMESİ VE KALİBRASYON
# ==========================================
print("\n3/5: Test seti üzerinde model performansı hesaplanıyor...")

y_pred = best_model.predict(X_test)
y_prob = best_model.predict_proba(X_test)[:, 1]

auc_score = roc_auc_score(y_test, y_prob)
brier = brier_score_loss(y_test, y_prob)
cm = confusion_matrix(y_test, y_pred)

print(f"\n--- MODEL METRİKLERİ ---")
print(f"ROC-AUC Skoru: {auc_score:.3f}")
print(f"Brier Skoru (Olasılık Kalibrasyonu - 0'a yakın olmalı): {brier:.3f}")
print("\nSınıflandırma Raporu:")
print(classification_report(y_test, y_pred, target_names=["Koşu", "Pas"]))
print("Karışıklık Matrisi (Confusion Matrix):")
print(f"TN: {cm[0, 0]} (Doğru Koşu) | FP: {cm[0, 1]} (Yanlış Pas)")
print(f"FN: {cm[1, 0]} (Yanlış Koşu) | TP: {cm[1, 1]} (Doğru Pas)")


# ==========================================
# 4. SHAP GLOBAL AÇIKLANABİLİRLİK (SUMMARY PLOT)
# ==========================================
print("\n4/5: SHAP TreeExplainer hesaplanıyor (Global Tells)...")

explainer = shap.TreeExplainer(best_model)
# Test setinden hesaplama süresini optimize etmek için 1500 örneklem
sample_indices = np.random.choice(len(X_test), size=min(1500, len(X_test)), replace=False)
X_shap = X_test.iloc[sample_indices]
shap_values = explainer(X_shap)

# Global Özellik Önem Grafiği (Beeswarm)
plt.figure(figsize=(10, 8))
shap.summary_plot(shap_values, X_shap, show=False, max_display=12)
plt.title("Snap Öncesi Oyunu Ele Veren En Güçlü Belirleyiciler (SHAP Summary)", fontsize=13, pad=15)
plt.tight_layout()
shap_summary_path = "output/shap_summary.png"
plt.savefig(shap_summary_path, dpi=300)
plt.close()
print(f"Global SHAP grafiği kaydedildi: {shap_summary_path}")


# ==========================================
# 5. POZİSYON BAZLI TEK OYUN ANALİZİ (WATERFALL PLOT)
# ==========================================
print("\n5/5: Örnek bir kritik oyun için yerel karar analizi üretiliyor...")

# Örnek: Test setinden kritik bir 3. Down oyunu seç
critical_idx = X_test[(X_test["down"] == 3) & (X_test["ydstogo"] <= 3)].index
target_idx = critical_idx[0] if len(critical_idx) > 0 else X_test.index[0]

single_shap = explainer(X_test.loc[[target_idx]])

plt.figure(figsize=(9, 6))
shap.plots.waterfall(single_shap[0], max_display=8, show=False)
plt.title(f"Pozisyon #{target_idx} - Snap Öncesi İpuçları (Waterfall)", fontsize=12, pad=15)
plt.tight_layout()
shap_waterfall_path = "output/shap_waterfall_example.png"
plt.savefig(shap_waterfall_path, dpi=300)
plt.close()
print(f"Örnek oyun Waterfall analizi kaydedildi: {shap_waterfall_path}")