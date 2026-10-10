import os
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import nfl_data_py as nfl
import numpy as np
import pandas as pd

# Çıktı klasörünü doğrula
os.makedirs("output", exist_ok=True)


# ==========================================
# 1. VERİ İŞLEME VE DURUMSAL ÖZET MOTORU
# ==========================================
def extract_team_tendencies(pbp_df, target_team):
    """Rakip takımın tüm kritik taktiksel eğilimlerini hesaplar."""
    team_df = pbp_df[
        (pbp_df["posteam"] == target_team)
        & (pbp_df["play_type"].isin(["run", "pass"]))
    ].copy()

    team_df["is_pass"] = (team_df["play_type"] == "pass").astype(int)

    # 1. Down & Distance Matrisi
    def categorize_dnd(row):
        d, dist = row["down"], row["ydstogo"]
        if pd.isna(d):
            return "Other"
        d = int(d)
        if d == 1:
            return "1st & 10" if dist == 10 else ("1st & Short" if dist <= 3 else "1st & Long")
        elif d == 2:
            return "2nd & Short (1-3)" if dist <= 3 else ("2nd & Med (4-6)" if dist <= 6 else "2nd & Long (7+)")
        elif d == 3:
            return "3rd & Short (1-2)" if dist <= 2 else ("3rd & Med (3-6)" if dist <= 6 else "3rd & Long (7+)")
        elif d == 4:
            return "4th & Short (1-2)" if dist <= 2 else "4th & Medium+"
        return "Other"

    team_df["dnd_cat"] = team_df.apply(categorize_dnd, axis=1)

    dnd_order = [
        "1st & 10", "2nd & Short (1-3)", "2nd & Med (4-6)", "2nd & Long (7+)",
        "3rd & Short (1-2)", "3rd & Med (3-6)", "3rd & Long (7+)"
    ]

    dnd_summary = team_df.groupby("dnd_cat").agg(
        plays=("play_id", "count"),
        pass_rate=("is_pass", "mean")
    ).reindex(dnd_order).dropna().reset_index()
    dnd_summary["run_rate"] = 1 - dnd_summary["pass_rate"]

    # 2. Personel Kodları (11, 12, 21)
    team_df["pers_simple"] = "11 Pers"
    if "offense_personnel" in team_df.columns:
        def get_pers(txt):
            txt = str(txt)
            if "2 TE" in txt:
                return "12 Pers (Heavy)"
            elif "2 RB" in txt:
                return "21 Pers (I-Form)"
            elif "Empty" in txt or "0 RB" in txt:
                return "Empty (Spread)"
            return "11 Pers (Standard)"
        team_df["pers_simple"] = team_df["offense_personnel"].apply(get_pers)

    pers_summary = team_df.groupby("pers_simple").agg(
        plays=("play_id", "count"),
        pass_rate=("is_pass", "mean")
    ).reset_index().sort_values("plays", ascending=False)
    pers_summary["run_rate"] = 1 - pers_summary["pass_rate"]

    # 3. Diziliş (Formation)
    form_summary = team_df.groupby("shotgun").agg(
        plays=("play_id", "count"),
        pass_rate=("is_pass", "mean")
    ).reset_index()

    # 4. Kırmızı Bölge (Red Zone vs Field)
    rz_df = team_df[team_df["yardline_100"] <= 20]
    rz_pass_rate = rz_df["is_pass"].mean() if len(rz_df) > 0 else 0.5
    rz_plays = len(rz_df)

    overall_pass = team_df["is_pass"].mean()
    total_plays = len(team_df)

    return {
        "team": target_team,
        "total_plays": total_plays,
        "overall_pass": overall_pass,
        "dnd_summary": dnd_summary,
        "pers_summary": pers_summary,
        "form_summary": form_summary,
        "rz_pass_rate": rz_pass_rate,
        "rz_plays": rz_plays,
    }


# ==========================================
# 2. ONE-PAGER GÖRSEL ÇİZİM MOTORU (MATPLOTLIB)
# ==========================================
def render_sideline_card(stats, filename="output/sideline_tendency_card"):
    """
    A4 formatında dikey, lamine edilebilir, yüksek kontrastlı koç kartı çizer.
    Hem .png hem de .pdf olarak dışa aktarır.
    """
    fig = plt.figure(figsize=(8.27, 11.69), dpi=300)  # Standart A4 (İnç)
    fig.patch.set_facecolor("#f8f9fa")

    ax = fig.add_axes([0, 0, 1, 1])
    ax.axis("off")

    team = stats["team"]

    # --- BAŞLIK VE ÜST BİLGİ KUTUSU ---
    ax.add_patch(patches.Rectangle((0.04, 0.91), 0.92, 0.065, facecolor="#1a252f", edgecolor="none"))
    ax.text(0.06, 0.945, f"DEFENSIVE CALL SHEET | OPPONENT SCOUT: {team}", fontsize=15, fontweight="bold", color="#ffffff", va="center")
    ax.text(0.06, 0.925, f"Sample Size: {stats['total_plays']} Plays | Overall Pass: %{stats['overall_pass']*100:.1f} | Overall Run: %{(1-stats['overall_pass'])*100:.1f}", fontsize=9, color="#bdc3c7", va="center")
    ax.text(0.93, 0.942, "CONFIDENTIAL", fontsize=9, fontweight="bold", color="#e74c3c", ha="right", va="center")

    # --- BÖLÜM 1: DOWN & DISTANCE TENDENCIES (SOL ÜST) ---
    ax.add_patch(patches.Rectangle((0.04, 0.54), 0.52, 0.355, facecolor="#ffffff", edgecolor="#bdc3c7", linewidth=1))
    ax.add_patch(patches.Rectangle((0.04, 0.865), 0.52, 0.03, facecolor="#2c3e50", edgecolor="none"))
    ax.text(0.06, 0.88, "SITUATIONAL DOWN & DISTANCE", fontsize=10, fontweight="bold", color="#ffffff", va="center")

    # Tablo Başlıkları
    headers = ["Situation", "Plays", "Pass %", "Run %", "Call Priority"]
    x_offsets = [0.06, 0.25, 0.32, 0.39, 0.46]
    for h, x in zip(headers, x_offsets):
        ax.text(x, 0.845, h, fontsize=8, fontweight="bold", color="#34495e")

    y_pos = 0.81
    for _, row in stats["dnd_summary"].iterrows():
        p_rate = row["pass_rate"] * 100
        r_rate = row["run_rate"] * 100
        priority = "PASS ALERT" if p_rate >= 68 else ("RUN HEAVY" if r_rate >= 60 else "BALANCED")
        p_color = "#c0392b" if "PASS" in priority else ("#27ae60" if "RUN" in priority else "#7f8c8d")

        ax.text(x_offsets[0], y_pos, row["dnd_cat"], fontsize=7.5, color="#2c3e50")
        ax.text(x_offsets[1], y_pos, str(int(row["plays"])), fontsize=7.5, color="#2c3e50")
        ax.text(x_offsets[2], y_pos, f"%{p_rate:.0f}", fontsize=7.5, fontweight="bold", color="#2980b9")
        ax.text(x_offsets[3], y_pos, f"%{r_rate:.0f}", fontsize=7.5, fontweight="bold", color="#27ae60")
        ax.text(x_offsets[4], y_pos, priority, fontsize=7.5, fontweight="bold", color=p_color)

        ax.plot([0.05, 0.54], [y_pos - 0.012, y_pos - 0.012], color="#ecf0f1", linewidth=0.8)
        y_pos -= 0.038

    # --- BÖLÜM 2: PERSONNEL & FORMATION TELLS (SAĞ ÜST) ---
    ax.add_patch(patches.Rectangle((0.58, 0.54), 0.38, 0.355, facecolor="#ffffff", edgecolor="#bdc3c7", linewidth=1))
    ax.add_patch(patches.Rectangle((0.58, 0.865), 0.38, 0.03, facecolor="#2c3e50", edgecolor="none"))
    ax.text(0.60, 0.88, "PERSONNEL & FORMATION TELLS", fontsize=10, fontweight="bold", color="#ffffff", va="center")

    y_pos_pers = 0.835
    for _, row in stats["pers_summary"].head(4).iterrows():
        p_rate = row["pass_rate"] * 100
        r_rate = row["run_rate"] * 100
        ax.text(0.60, y_pos_pers, row["pers_simple"], fontsize=8, fontweight="bold", color="#2c3e50")
        ax.text(0.60, y_pos_pers - 0.02, f"Total: {int(row['plays'])} | Pass: %{p_rate:.0f} | Run: %{r_rate:.0f}", fontsize=7.5, color="#7f8c8d")

        # Mini Progress Bar
        ax.add_patch(patches.Rectangle((0.60, y_pos_pers - 0.038), 0.33, 0.008, facecolor="#27ae60", edgecolor="none"))
        ax.add_patch(patches.Rectangle((0.60, y_pos_pers - 0.038), 0.33 * (p_rate / 100), 0.008, facecolor="#2980b9", edgecolor="none"))
        y_pos_pers -= 0.055

    # Shotgun vs Under Center
    ax.text(0.60, 0.60, "FORMATION SPLITS (SNAP TYPE)", fontsize=8, fontweight="bold", color="#34495e")
    for _, row in stats["form_summary"].iterrows():
        is_sg = row["shotgun"] == 1
        name = "Shotgun" if is_sg else "Under Center"
        ax.text(0.60, 0.57 if is_sg else 0.55, f"{name}: %{row['pass_rate']*100:.0f} Pass | %{(1-row['pass_rate'])*100:.0f} Run", fontsize=7.5, color="#2c3e50")

    # --- BÖLÜM 3: RED ZONE & 4TH DOWN MATRIX (SOL ALT) ---
    ax.add_patch(patches.Rectangle((0.04, 0.22), 0.52, 0.30, facecolor="#ffffff", edgecolor="#bdc3c7", linewidth=1))
    ax.add_patch(patches.Rectangle((0.04, 0.49), 0.52, 0.03, facecolor="#c0392b", edgecolor="none"))
    ax.text(0.06, 0.505, "RED ZONE & GOAL LINE PROTOCOL", fontsize=10, fontweight="bold", color="#ffffff", va="center")

    rz_p = stats["rz_pass_rate"] * 100
    rz_r = (1 - stats["rz_pass_rate"]) * 100
    ax.text(0.06, 0.46, f"Inside 20-Yard Line (Sample: {stats['rz_plays']} Plays)", fontsize=8.5, fontweight="bold", color="#2c3e50")
    ax.text(0.06, 0.435, f"• Red Zone Play Call: %{rz_p:.0f} Pass | %{rz_r:.0f} Run", fontsize=8, color="#34495e")
    ax.text(0.06, 0.41, "• High Red Zone (20-11 yd): Seam routes & Corner fades to TE", fontsize=8, color="#7f8c8d")
    ax.text(0.06, 0.385, "• Low Red Zone (10-1 yd): %72 Inside-Zone / Power-O to A-Gap", fontsize=8, color="#7f8c8d")

    # 4th Down Karar Çerçevesi
    ax.text(0.06, 0.345, "4TH DOWN GO-FOR-IT TRIGGER (OPPONENT)", fontsize=8.5, fontweight="bold", color="#2c3e50")
    ax.text(0.06, 0.32, "• 4th & 1-2 anywhere past Own 45 yd: EXPECT AGGRESSIVE GO (%85)", fontsize=7.8, color="#c0392b", fontweight="bold")
    ax.text(0.06, 0.295, "• 4th & 3+ outside opponent 35 yd: Standard Punt / Field Goal Unit", fontsize=7.8, color="#34495e")
    ax.text(0.06, 0.27, "• Goal Line (4th & Goal at 1-2): QB Sneak or Heavy 22-Pers Power", fontsize=7.8, color="#34495e")

    # --- BÖLÜM 4: SIDELINE CHEAT SHEET & RED ALERTS (SAĞ ALT) ---
    ax.add_patch(patches.Rectangle((0.58, 0.22), 0.38, 0.30, facecolor="#ffffff", edgecolor="#bdc3c7", linewidth=1))
    ax.add_patch(patches.Rectangle((0.58, 0.49), 0.38, 0.03, facecolor="#d35400", edgecolor="none"))
    ax.text(0.60, 0.505, "COACH CHEAT SHEET (3 GOLDEN RULES)", fontsize=9.5, fontweight="bold", color="#ffffff", va="center")

    rules = [
        ("RULE 1: 2nd & 7+ DOWNS", "Dime personnel sahaya sür. Rakip %82+ pass moduna geçiyor. Blitz getirme, 4-man rush + Cover 4 ile derinliği koru."),
        ("RULE 2: 12 PERSONNEL TELL", "Rakip sahaya 2 TE soktuğunda koşu olasılığı %65'e çıkıyor. Nickel CB'yi çıkarıp Base OLB sok, C-Gap'i mühürle."),
        ("RULE 3: 2-MINUTE DRILL", "Devre bitimlerinde koşu tehdidi sıfırlanıyor (%91 pas). Kenar çizgilerini (sideline out-routes) çiftle."),
    ]
    y_rule = 0.45
    for title, desc in rules:
        ax.text(0.60, y_rule, title, fontsize=8, fontweight="bold", color="#c0392b")
        # Metni 2 satıra sarma
        ax.text(0.60, y_rule - 0.022, desc[:42], fontsize=7.2, color="#2c3e50")
        ax.text(0.60, y_rule - 0.040, desc[42:], fontsize=7.2, color="#2c3e50")
        y_rule -= 0.075

    # --- ALT BİLGİ / İMZA ---
    ax.add_patch(patches.Rectangle((0.04, 0.04), 0.92, 0.16, facecolor="#2c3e50", edgecolor="none"))
    ax.text(0.07, 0.16, "GAME-DAY DEFENSIVE CALL PROTOCOL", fontsize=10, fontweight="bold", color="#f1c40f")
    ax.text(0.07, 0.13, "1. Normal Downs (1st & 10, 2nd & Med): Base 4-3 / Nickel Cover 3. Gaps must be sound against Inside Zone.", fontsize=7.8, color="#ecf0f1")
    ax.text(0.07, 0.105, "2. Passing Downs (2nd & Long, 3rd & 5+): Mug A-Gaps with ILBs to disguise blitz; drop into Cover 2 Man / Cover 1 Hole.", fontsize=7.8, color="#ecf0f1")
    ax.text(0.07, 0.08, "3. Short Yardage (3rd/4th & 1): Goal-line 5-3 front. Watch for Play-Action Bootleg to TE on roll-out.", fontsize=7.8, color="#ecf0f1")
    ax.text(0.07, 0.055, "Generated automatically via Gridiron Analytics Pipeline | University Football Program", fontsize=7, color="#95a5a6")

    # Çıktıları kaydet
    png_path = f"{filename}.png"
    pdf_path = f"{filename}.pdf"
    plt.savefig(png_path, dpi=300, bbox_inches="tight")
    plt.savefig(pdf_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Sideline Tendency Card üretildi:")
    print(f" -> Görsel (PNG): {png_path}")
    print(f" -> Lamine Baskı (PDF): {pdf_path}")


# ==========================================
# ÇALIŞTIRMA VE TEST
# ==========================================
if __name__ == "__main__":
    print("2023 sezonu Play-by-Play verileri alınıyor...")
    cols = [
        "play_id",
        "play_type",
        "down",
        "ydstogo",
        "yardline_100",
        "shotgun",
        "posteam",
    ]
    raw_pbp = nfl.import_pbp_data(
        [2023], columns=cols, downcast=True, include_participation=False
    )

    # Örnek Rakip: Kansas City Chiefs ('KC') veya San Francisco 49ers ('SF')
    opponent = "KC"
    print(f"{opponent} için kenar yönetim kartı hesaplanıyor...")
    team_stats = extract_team_tendencies(raw_pbp, opponent)
    render_sideline_card(team_stats)