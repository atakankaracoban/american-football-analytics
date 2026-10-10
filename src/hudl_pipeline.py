import os
import re

import numpy as np
import pandas as pd


# ==========================================
# 1. SAHA KONUMU STANDARTLAŞTIRICI
# ==========================================
def parse_hudl_yardline(yard_val):
    """
    Hudl'ın karmaşık saha konumu formatlarını 'yardline_100' değerine dönüştürür.
    (100 = Kendi endzone'u, 50 = Orta saha, 0 = Rakip endzone)
    Örnekler:
      - '-35' veya 'OWN 35' -> 65
      - '+20' veya 'OPP 20' -> 20
      - '50' -> 50
    """
    if pd.isna(yard_val):
        return np.nan

    txt = str(yard_val).strip().upper()

    # 50 Yard (Orta Saha)
    if txt in ["50", "50.0"]:
        return 50.0

    # '+20' veya '-35' formatı
    if txt.startswith("+"):
        try:
            return float(txt.replace("+", ""))
        except ValueError:
            pass
    elif txt.startswith("-"):
        try:
            return 100.0 - float(txt.replace("-", ""))
        except ValueError:
            pass

    # 'OPP 25' veya 'OWN 40' formatı
    if "OPP" in txt:
        nums = re.findall(r"\d+", txt)
        if nums:
            return float(nums[0])
    elif "OWN" in txt:
        nums = re.findall(r"\d+", txt)
        if nums:
            return 100.0 - float(nums[0])

    # Yalnızca düz sayı girilmişse (<50 ise rakip veya kendi varsayımı)
    nums = re.findall(r"\d+", txt)
    if nums:
        val = float(nums[0])
        # Hudl'da genellikle 50 altı rakamlar girildiğinde orta saha altı varsayılır
        return 100.0 - val if val <= 50 else val

    return np.nan


# ==========================================
# 2. OYUN TİPİ VE SONUÇ AYRIŞTIRICI
# ==========================================
def standardize_play_type(row):
    """
    Hudl 'PLAY TYPE' ve 'RESULT' açıklamalarından net 'run' veya 'pass' çıkarır.
    Sack durumları modern analitik standardına uygun olarak 'pass' kabul edilir.
    """
    pt = str(row.get("PLAY TYPE", "")).lower()
    res = str(row.get("RESULT", "")).lower()
    desc = str(row.get("PLAY DESCRIPTION", "")).lower()

    combined = f"{pt} {res} {desc}"

    if any(
        k in combined
        for k in ["punt", "field goal", "extra point", "kickoff", "pat"]
    ):
        return "special_teams"
    elif any(
        k in combined
        for k in ["sack", "interception", "pass", "incomplete", "completion"]
    ):
        return "pass"
    elif any(k in combined for k in ["rush", "run", "carry", "keeper", "scramble"]):
        return "run"
    elif "run" in pt:
        return "run"
    elif "pass" in pt:
        return "pass"
    return "other"


# ==========================================
# 3. ANA HUDL TEMİZLEME MOTORU
# ==========================================
def clean_hudl_csv(filepath):
    """Ham Hudl CSV dökümünü analiz motorlarımızla %100 uyumlu hale getirir."""
    print(f"Hudl dökümü okunuyor: {filepath}")
    raw_df = pd.read_csv(filepath)

    # Kolon adlarını normalize et (Büyük harf ve boşluksuz)
    raw_df.columns = [c.strip().upper() for c in raw_df.columns]

    df = pd.DataFrame()

    # Oyun Kimlikleri
    df["play_id"] = np.arange(1, len(raw_df) + 1)
    df["game_id"] = raw_df.get("GAME", "COLLEGE_GAME_1")
    df["posteam"] = raw_df.get("OFF TEAM", raw_df.get("OFFENSE", "OFF"))
    df["defteam"] = raw_df.get("DEF TEAM", raw_df.get("DEFENSE", "DEF"))

    # Çeyrek & Zaman
    df["qtr"] = (
        pd.to_numeric(raw_df.get("QTR", 1), errors="coerce").fillna(1).astype(int)
    )

    # Down & Distance
    down_col = next((c for c in ["DN", "DOWN"] if c in raw_df.columns), None)
    df["down"] = (
        pd.to_numeric(raw_df[down_col], errors="coerce")
        if down_col
        else np.nan
    )

    dist_col = next(
        (c for c in ["DIST", "DISTANCE", "YDS TO GO"] if c in raw_df.columns),
        None,
    )
    df["ydstogo"] = (
        pd.to_numeric(raw_df[dist_col], errors="coerce")
        if dist_col
        else np.nan
    )

    # Saha Pozisyonu (Yardline 100)
    yd_col = next(
        (c for c in ["YARD LN", "YARDLINE", "BALL ON"] if c in raw_df.columns),
        None,
    )
    if yd_col:
        df["yardline_100"] = raw_df[yd_col].apply(parse_hudl_yardline)
    else:
        df["yardline_100"] = 50.0

    # "Goal to go" durumunda mesafe düzeltmesi
    df.loc[
        df["ydstogo"].isna() & (df["yardline_100"] <= 10), "ydstogo"
    ] = df["yardline_100"]

    # Kazanılan Yard (GN/LS)
    gain_col = next(
        (
            c
            for c in ["GN/LS", "GAIN/LOSS", "YARDS", "YDS GAINED"]
            if c in raw_df.columns
        ),
        None,
    )
    df["yards_gained"] = (
        pd.to_numeric(raw_df[gain_col], errors="coerce").fillna(0)
        if gain_col
        else 0
    )

    # Oyun Tipi Sınıflandırma
    df["play_type"] = raw_df.apply(standardize_play_type, axis=1)

    # Diziliş ve Hash Mark
    form_col = next(
        (c for c in ["OFF FORM", "FORMATION"] if c in raw_df.columns), None
    )
    df["formation"] = (
        raw_df[form_col].astype(str).str.upper().fillna("UNKNOWN")
        if form_col
        else "UNKNOWN"
    )
    df["shotgun"] = df["formation"].apply(
        lambda x: 1 if "GUN" in x or "PISTOL" in x else 0
    )

    hash_col = next((c for c in ["HASH"] if c in raw_df.columns), None)
    if hash_col:
        hash_mapping = {
            "L": "Left",
            "R": "Right",
            "M": "Middle",
            "LEFT": "Left",
            "RIGHT": "Right",
            "MIDDLE": "Middle",
        }
        df["hash"] = (
            raw_df[hash_col]
            .astype(str)
            .str.upper()
            .map(hash_mapping)
            .fillna("Middle")
        )
    else:
        df["hash"] = "Middle"

    # Personel Grubu (11, 12, 21 vb.)
    pers_col = next((c for c in ["OFF PER", "PERSONNEL"] if c in raw_df.columns), None)
    if pers_col:
        def clean_pers(val):
            val_str = str(val).strip()
            digits = re.findall(r"\d+", val_str)
            if len(digits) >= 2:
                return f"{digits[0]}{digits[1]}"
            elif len(digits) == 1 and len(digits[0]) == 2:
                return digits[0]
            return "11"  # Standart varsayılan

        df["personnel_code"] = raw_df[pers_col].apply(clean_pers)
        df["offense_personnel"] = df["personnel_code"].apply(
            lambda x: (
                f"{x[0]} RB, {x[1]} TE, "
                f"{5 - int(x[0]) - int(x[1])} WR"
                if len(x) == 2
                else "1 RB, 1 TE, 3 WR"
            )
        )
    else:
        df["personnel_code"] = "11"
        df["offense_personnel"] = "1 RB, 1 TE, 3 WR"

    # Koşu Koridoru & Pas Detayları (Varsa)
    gap_col = next(
        (c for c in ["RUN GAP", "GAP", "HOLE"] if c in raw_df.columns), None
    )
    df["run_gap"] = (
        raw_df[gap_col].astype(str).str.lower().fillna("middle")
        if gap_col
        else "middle"
    )

    # Bağlamsal Success Rate Hesabı
    c1 = (df["down"] == 1) & (df["yards_gained"] >= 0.50 * df["ydstogo"])
    c2 = (df["down"] == 2) & (df["yards_gained"] >= 0.70 * df["ydstogo"])
    c3 = (df["down"].isin([3, 4])) & (df["yards_gained"] >= df["ydstogo"])
    df["is_success"] = (c1 | c2 | c3).astype(int)

    # Eksik verileri filtrele (Sadece scrimmage oyunları)
    clean_plays = df[
        df["play_type"].isin(["run", "pass"])
        & df["down"].notna()
        & df["ydstogo"].notna()
        & df["yardline_100"].notna()
    ].copy()

    clean_plays["is_pass"] = (clean_plays["play_type"] == "pass").astype(int)

    return clean_plays


# ==========================================
# 4. TEST VE ÖRNEK VERİ ÜRETİCİ
# ==========================================
def generate_sample_hudl_csv(filepath="data/sample_hudl_export.csv"):
    """Eğer takımın henüz CSV dosyası yoksa test için örnek Hudl dökümü oluşturur."""
    os.makedirs("data", exist_ok=True)
    n_plays = 65

    np.random.seed(42)
    downs = np.random.choice([1, 2, 3, 4], size=n_plays, p=[0.42, 0.33, 0.20, 0.05])
    dists = [10 if d == 1 else np.random.randint(1, 12) for d in downs]
    yard_prefixes = np.random.choice(["-", "+", "OWN ", "OPP "], size=n_plays)
    yard_nums = np.random.randint(10, 48, size=n_plays)
    yardlines = [f"{p}{n}" for p, n in zip(yard_prefixes, yard_nums)]

    play_types = np.random.choice(
        ["Pass", "Rush", "Pass Complete", "Sack", "Rush - Inside"],
        size=n_plays,
    )
    gains = [np.random.randint(-4, 18) for _ in range(n_plays)]
    forms = np.random.choice(
        ["GUN TRIPS RT", "GUN 2X2", "PISTOL HEAVY", "UNDER I-FORM"],
        size=n_plays,
    )
    pers = np.random.choice(
        ["11", "12", "21", "10"],
        size=n_plays,
        p=[0.55, 0.25, 0.15, 0.05],
    )
    hashes = np.random.choice(["L", "R", "M"], size=n_plays)

    sample_df = pd.DataFrame(
        {
            "GAME": "WEEK_1_OPPONENT",
            "OFF TEAM": "RIVAL_UNIV",
            "DEF TEAM": "OUR_TEAM",
            "QTR": np.random.choice([1, 2, 3, 4], size=n_plays),
            "DN": downs,
            "DIST": dists,
            "YARD LN": yardlines,
            "PLAY TYPE": play_types,
            "GN/LS": gains,
            "OFF FORM": forms,
            "OFF PER": pers,
            "HASH": hashes,
            "RESULT": [
                "Complete" if "Pass" in pt else "Tackle" for pt in play_types
            ],
        }
    )

    sample_df.to_csv(filepath, index=False)
    print(f"Test için sentetik Hudl dosyası oluşturuldu: {filepath}")


# ==========================================
# ÇALIŞTIRMA VE DOĞRULAMA
# ==========================================
if __name__ == "__main__":
    os.makedirs("data", exist_ok=True)
    raw_csv_path = "data/sample_hudl_export.csv"

    # Gerçek dosya yoksa sentetik dosya üret
    if not os.path.exists(raw_csv_path):
        generate_sample_hudl_csv(raw_csv_path)

    # Veriyi temizle ve pipeline'dan geçir
    cleaned_df = clean_hudl_csv(raw_csv_path)

    # Çıktıları kaydet
    out_csv = "data/cleaned_hudl_pbp.csv"
    out_parquet = "data/cleaned_hudl_pbp.parquet"
    cleaned_df.to_csv(out_csv, index=False)
    cleaned_df.to_parquet(out_parquet, index=False)

    print("\n--- TEMİZLENMİŞ HUDL VERİSİ ÖZETİ ---")
    print(f"Toplam İşlenen Hücum Oyunu: {len(cleaned_df)}")
    print(f"Genel Pas Oranı: %{cleaned_df['is_pass'].mean() * 100:.1f}")
    print(
        f"Genel Başarı Oranı (Success Rate): "
        f"%{cleaned_df['is_success'].mean() * 100:.1f}"
    )
    print(f"Kaydedilen Dosyalar: {out_csv} ve {out_parquet}")
    print("\nİlk 3 Oyunun Temizlenmiş Şeması:")
    preview_columns = [
        "play_id",
        "down",
        "ydstogo",
        "yardline_100",
        "play_type",
        "formation",
        "personnel_code",
        "is_success",
    ]
    print(cleaned_df[preview_columns].head(3).to_string(index=False))