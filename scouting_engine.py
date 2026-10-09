import nfl_data_py as nfl
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

# 1. VERİ ÇEKME (2023 sezonu play-by-play verisi)
print("Veri yükleniyor...")
cols = [
    'game_id', 'play_id', 'play_type', 'down', 'ydstogo', 'yardline_100',
    'quarter_seconds_remaining', 'qtr', 'score_differential', 'shotgun',
    'no_huddle', 'pass_length', 'pass_location', 'run_location', 'run_gap',
    'posteam', 'defteam'
]
pbp = nfl.import_pbp_data([2023], columns=cols, downcast=True)

# 2. VERİ TEMİZLEME & FİLTRELEME
# Yalnızca standart hücum oyunları (Koşu ve Pas)
clean_plays = pbp[pbp['play_type'].isin(['run', 'pass'])].dropna(
    subset=['down', 'ydstogo', 'yardline_100']
).copy()

# Hedef Değişken: 1 = Pas, 0 = Koşu
clean_plays['is_pass'] = (clean_plays['play_type'] == 'pass').astype(int)

# 3. DURUMSAL EĞİLİM ANALİZİ (Tendency Extraction)
def get_down_distance_tendencies(df, team):
    team_plays = df[df['posteam'] == team].copy()
    
    # Durum kategorileri
    def categorize_situation(row):
        d, dist = row['down'], row['ydstogo']
        if dist <= 2: return f"{int(d)}rd & Short (<=2)"
        elif dist <= 6: return f"{int(d)}rd & Medium (3-6)"
        else: return f"{int(d)}rd & Long (7+)"

    team_plays['situation'] = team_plays.apply(categorize_situation, axis=1)
    
    summary = team_plays.groupby('situation').agg(
        total_plays=('play_type', 'count'),
        pass_rate=('is_pass', 'mean')
    ).reset_index()
    
    summary['run_rate'] = 1 - summary['pass_rate']
    summary['pass_rate'] = (summary['pass_rate'] * 100).round(1)
    summary['run_rate'] = (summary['run_rate'] * 100).round(1)
    return summary.sort_values(by='total_plays', ascending=False)

# Örnek bir takım için eğilim raporu
sample_tendencies = get_down_distance_tendencies(clean_plays, 'KC')
print("\n--- ÖRNEK EĞİLİM TABLOSU (Kansas City Chiefs) ---")
print(sample_tendencies.head(10).to_string(index=False))

# 4. PREDICTIVE MODEL (Snap Öncesi Run/Pass Sınıflandırma)
features = [
    'down', 'ydstogo', 'yardline_100', 'quarter_seconds_remaining',
    'qtr', 'score_differential', 'shotgun', 'no_huddle'
]
X = clean_plays[features]
y = clean_plays['is_pass']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

model = XGBClassifier(
    n_estimators=100,
    max_depth=4,
    learning_rate=0.08,
    eval_metric='logloss',
    random_state=42
)
model.fit(X_train, y_train)

# 5. DEĞERLENDİRME
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

print("\n--- MODEL BAŞARISI (Run vs Pass) ---")
print(f"ROC-AUC Skoru: {roc_auc_score(y_test, y_prob):.3f}")
print(classification_report(y_test, y_pred, target_names=['Koşu', 'Pas']))

# Özellik Önem Düzeyleri (Feature Importance)
importance = pd.DataFrame({
    'Feature': features,
    'Importance': model.feature_importances_
}).sort_values('Importance', ascending=False)

print("\n--- SNAP ÖNCESİ BELİRLEYİCİ DEĞİŞKENLER ---")
print(importance.to_string(index=False))