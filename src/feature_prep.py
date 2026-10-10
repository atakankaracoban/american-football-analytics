from pathlib import Path
import re
import numpy as np
import pandas as pd
import nfl_data_py as nfl


# ==========================================
# 1. PERSONEL AYRIŞTIRMA (PERSONNEL PARSER)
# ==========================================
def parse_personnel(df):
    """
    '1 RB, 1 TE, 3 WR' gibi metinleri ayrıştırarak:
    - n_rb, n_te, n_wr sayılarını çıkarır.
    - Amerikan futbolu standardı 2 haneli personel kodunu üretir (Örn: 11, 12, 21).
    """
    df = df.copy()

    def extract_counts(text):
        if pd.isna(text):
            return 1, 1, 3  # Standart varsayılan (11 Personnel)

        rb_match = re.search(r'(\d+)\s*RB', str(text))
        te_match = re.search(r'(\d+)\s*TE', str(text))
        wr_match = re.search(r'(\d+)\s*WR', str(text))

        rb = int(rb_match.group(1)) if rb_match else 0
        te = int(te_match.group(1)) if te_match else 0
        wr = int(wr_match.group(1)) if wr_match else 0
        return rb, te, wr

    personnel = df.get(
        'offense_personnel', pd.Series(pd.NA, index=df.index, dtype='string')
    )
    counts = personnel.apply(extract_counts)
    df['n_rb'] = [c[0] for c in counts]
    df['n_te'] = [c[1] for c in counts]
    df['n_wr'] = [c[2] for c in counts]

    # İki basamaklı standart personel kodu: (RB Sayısı * 10) + TE Sayısı
    # Örn: 1 RB + 1 TE = '11', 1 RB + 2 TE = '12', 2 RB + 1 TE = '21'
    df['personnel_code'] = (df['n_rb'].astype(str) + df['n_te'].astype(str))

    # Yaygın olmayan gruplamaları 'Other' altında toplama
    top_personnels = ['11', '12', '21', '22', '10', '13', '20']
    df['personnel_code'] = df['personnel_code'].apply(
        lambda x: x if x in top_personnels else 'Other'
    )
    return df


# ==========================================
# 2. DURUMSAL VE SAHA BAĞLAMI DEĞİŞKENLERİ
# ==========================================
def engineer_situational_features(df):
    """
    Snap öncesi oyun bağlamını, zaman baskısını ve saha pozisyonunu modeller.
    """
    df = df.copy()

    # Saha Bölgeleri (Field Zones)
    # yardline_100: Rakip endzone'a olan mesafe (100 = kendi endzone çizgisi)
    def categorize_field_zone(yd):
        if yd >= 90:
            return 'Backed_Up'      # Kendi 10 yardı ve gerisi (Hata payı sıfır)
        elif yd > 50:
            return 'Own_Territory'  # Kendi yarı sahası
        elif yd > 20:
            return 'Opp_Territory'  # Rakip yarı sahası
        elif yd > 5:
            return 'Red_Zone'       # 20 ile 5 yard arası
        else:
            return 'Goal_Line'      # 5 yard ve içi

    df['field_zone'] = df['yardline_100'].apply(categorize_field_zone)

    # Down & Mesafe Kombinasyonları
    df['is_short_yardage'] = (df['ydstogo'] <= 2).astype(int)
    df['is_long_yardage'] = (df['ydstogo'] >= 7).astype(int)
    df['is_3rd_or_4th_down'] = df['down'].isin([3, 4]).astype(int)

    # Zaman Baskısı ve Skor Bağlamı
    # 2-Minute Drill: Devre veya maç sonuna 2 dakikadan az kalması
    df['is_two_minute_drill'] = (
        (df['half_seconds_remaining'] <= 120) & (df['half_seconds_remaining'] > 0)
    ).astype(int)

    # Çöp Zaman (Garbage Time): 4. çeyrekte 16+ sayı fark olması
    df['is_garbage_time'] = (
        (df['qtr'] == 4) & (df['score_differential'].abs() >= 16)
    ).astype(int)

    # Etkileşim Terimleri (Cross Features)
    # Koçlar için kritik: Shotgun dizilişinde kısa mesafe mi, uzun mesafe mi?
    df['shotgun_x_ydstogo'] = df['shotgun'] * df['ydstogo']
    df['down_x_ydstogo'] = df['down'] * df['ydstogo']

    return df


# ==========================================
# 3. FORMASYON VE HUDL UYUMLULUK ENTEGRASYONU
# ==========================================
def handle_formations_and_hash(df):
    """
    Dizilişleri standartlaştırır ve üniversite/Hudl dökümlerinde bulunan
    hash mark (Field/Boundary) verilerini modele hazırlar.
    """
    df = df.copy()

    # Formasyon temizleme
    if 'offense_formation' in df.columns:
        df['formation'] = df['offense_formation'].fillna('SHOTGUN')
        # Nadir formasyonları grupla
        valid_formations = ['SHOTGUN', 'SINGLEBACK', 'I_FORM', 'PISTOL', 'EMPTY', 'UNDER CENTER']
        df['formation'] = df['formation'].apply(
            lambda x: x if x in valid_formations else 'OTHER'
        )
    else:
        # Hudl'da yalnızca shotgun ikili sütunu varsa
        df['formation'] = np.where(df['shotgun'] == 1, 'SHOTGUN', 'UNDER_CENTER')

    # Hash Mark Simülasyonu / Hudl Eşlemesi
    # Üniversite verisinde 'hash' (Left, Right, Middle) sütunu varsa Field/Boundary türetilir.
    if 'hash' not in df.columns:
        # NFL verisinde açık hash mark sütunu yoksa, sentetik/varsayılan kategori
        df['hash_position'] = 'Middle'
    else:
        df['hash_position'] = df['hash'].fillna('Middle')

    return df


# ==========================================
# 4. KODLAMA VE EĞİTİM VERİ SETİ HAZIRLIĞI
# ==========================================
def prepare_pre_snap_dataset(pbp_df):
    """
    Tüm boru hattını birleştirir, kategorik değişkenleri One-Hot encode eder
    ve yalnızca snap öncesi özellikleri döndürür.
    """
    # 1. Yalnızca Koşu ve Pas oyunları
    df = pbp_df[pbp_df['play_type'].isin(['run', 'pass'])].dropna(
        subset=['down', 'ydstogo', 'yardline_100', 'score_differential']
    ).copy()

    # Hedef Değişken (Target): 1 = Pas, 0 = Koşu
    df['target_is_pass'] = (df['play_type'] == 'pass').astype(int)

    # 2. Özellik mühendisliği adımları
    df = parse_personnel(df)
    df = engineer_situational_features(df)
    df = handle_formations_and_hash(df)

    # 3. Sayısal ve Kategorik Değişken Listesi
    numeric_features = [
        'down', 'ydstogo', 'yardline_100', 'qtr',
        'half_seconds_remaining', 'score_differential',
        'shotgun', 'no_huddle', 'n_rb', 'n_te', 'n_wr',
        'is_short_yardage', 'is_long_yardage', 'is_3rd_or_4th_down',
        'is_two_minute_drill', 'is_garbage_time',
        'shotgun_x_ydstogo', 'down_x_ydstogo'
    ]

    categorical_features = ['personnel_code', 'field_zone', 'formation', 'hash_position']

    # One-Hot Encoding
    encoded_cats = pd.get_dummies(
        df[categorical_features],
        prefix=['pers', 'zone', 'form', 'hash'],
        drop_first=True,
        dtype=int
    )

    # Nihai Veri Seti
    X = pd.concat([df[numeric_features], encoded_cats], axis=1)
    y = df['target_is_pass']

    metadata = df[['game_id', 'play_id', 'posteam', 'defteam']]

    return X, y, metadata


# ==========================================
# TEST VE KONTROL
# ==========================================
if __name__ == '__main__':
    print("Test verisi çekiliyor (2023 Sezonu)...")
    cols = [
        'game_id', 'play_id', 'play_type', 'down', 'ydstogo', 'yardline_100',
        'qtr', 'half_seconds_remaining', 'score_differential', 'shotgun',
        'no_huddle', 'posteam', 'defteam'
    ]
    raw_pbp = nfl.import_pbp_data(
        [2023], columns=cols, downcast=True, include_participation=False
    )

    X, y, meta = prepare_pre_snap_dataset(raw_pbp)

    print("\n--- ÖZELLİK MATRİSİ ÖZETİ ---")
    print(f"Toplam Oyun Sayısı: {X.shape[0]}")
    print(f"Türetilen Snap Öncesi Özellik Sayısı: {X.shape[1]}")
    print(f"Pas Oranı (Sınıf Dengesi): %{y.mean() * 100:.1f}")
    print("\nİlk 5 Satır Özellik Başlıkları:")
    print(list(X.columns[:15]))

    output_dir = Path(__file__).resolve().parents[1] / "output"
    output_dir.mkdir(parents=True, exist_ok=True)
    X.to_csv(output_dir / "pre_snap_features.csv", index=False)
    y.rename("target_is_pass").to_frame().to_csv(
        output_dir / "pre_snap_target.csv", index=False
    )
    meta.to_csv(output_dir / "pre_snap_metadata.csv", index=False)
    print(f"\nCSV çıktıları kaydedildi: {output_dir}")