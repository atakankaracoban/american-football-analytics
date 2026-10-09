from pathlib import Path

import matplotlib.pyplot as plt
import nfl_data_py as nfl
import numpy as np
import pandas as pd
import seaborn as sns

# Görsel stil ayarları
sns.set_theme(style="whitegrid")
plt.rcParams["font.sans-serif"] = "DejaVu Sans"
plt.rcParams["axes.edgecolor"] = "#cccccc"


# 1. VERİ ZENGİNLEŞTİRME: SUCCESS RATE HESABI
def enrich_pbp_data(df):
    """Her oyun için bağlamsal Success Rate ve etiketleri ekler."""
    df = df.copy()

    # Success Rate Şartları:
    # 1st down: Gereken mesafenin >= %50'si
    # 2nd down: Gereken mesafenin >= %70'i
    # 3rd & 4th down: 1st down almak (>= %100)
    cond_1st = (df["down"] == 1) & (df["yards_gained"] >= 0.50 * df["ydstogo"])
    cond_2nd = (df["down"] == 2) & (df["yards_gained"] >= 0.70 * df["ydstogo"])
    cond_late = (df["down"].isin([3, 4])) & (
        df["yards_gained"] >= df["ydstogo"]
    )

    df["is_success"] = (cond_1st | cond_2nd | cond_late).astype(int)

    # Koşu Koridoru standardizasyonu (7 Koridor)
    def map_run_gap(row):
        if row["run_location"] == "middle":
            return "Center/A-Gap"
        elif pd.notna(row["run_location"]) and pd.notna(row["run_gap"]):
            loc = "L" if row["run_location"] == "left" else "R"
            gap = row["run_gap"].capitalize()
            return f"{loc} {gap}"
        return "Unknown"

    df["gap_detail"] = df.apply(map_run_gap, axis=1)
    return df


# 2. PASSING CHART (6 BÖLGELİ PAS ISI HARİTASI)
def plot_passing_chart(df, team, save_path="passing_chart.png"):
    """Sahanın 6 ana bölgesindeki pas hacmini ve isabet oranını görselleştirir."""
    pass_df = df[
        (df["posteam"] == team)
        & (df["play_type"] == "pass")
        & (df["pass_length"].isin(["short", "deep"]))
        & (df["pass_location"].isin(["left", "middle", "right"]))
    ].copy()

    # Pivot: Toplam pas sayısı ve isabet yüzdesi
    summary = (
        pass_df.groupby(["pass_length", "pass_location"])
        .agg(
            attempts=("play_id", "count"),
            completions=("complete_pass", "sum"),
            avg_epa=("epa", "mean"),
        )
        .reset_index()
    )

    summary["comp_pct"] = (
        (summary["completions"] / summary["attempts"]) * 100
    ).round(1)

    # 2x3 Matris formatına dönüştürme
    matrix_attempts = summary.pivot(
        index="pass_length", columns="pass_location", values="attempts"
    ).reindex(index=["deep", "short"], columns=["left", "middle", "right"])
    matrix_comp = summary.pivot(
        index="pass_length", columns="pass_location", values="comp_pct"
    ).reindex(index=["deep", "short"], columns=["left", "middle", "right"])

    # Etiket formatı: "12/15 (%80)"
    labels = np.empty(matrix_attempts.shape, dtype=object)
    for i in range(matrix_attempts.shape[0]):
        for j in range(matrix_attempts.shape[1]):
            att = matrix_attempts.iloc[i, j]
            comp = matrix_comp.iloc[i, j]
            if pd.notna(att) and att > 0:
                labels[i, j] = f"{int(att)} Pas\n%{comp:.0f} İsabet"
            else:
                labels[i, j] = "Veri Yok"

    fig, ax = plt.subplots(figsize=(9, 6))
    sns.heatmap(
        matrix_attempts,
        annot=labels,
        fmt="",
        cmap="Blues",
        cbar_kws={"label": "Pas Denemesi Sayısı"},
        linewidths=2,
        linecolor="#ffffff",
        ax=ax,
    )

    ax.set_title(
        f"{team} - Pas Dağılımı ve İsabet Bölgeleri (Passing Chart)",
        fontsize=14,
        fontweight="bold",
        pad=15,
    )
    ax.set_xlabel("Pas Yönü", fontsize=12, labelpad=10)
    ax.set_ylabel("Pas Derinliği", fontsize=12)
    ax.set_xticklabels(["Sol (Left)", "Orta (Middle)", "Sağ (Right)"])
    ax.set_yticklabels(
        ["Derin (15+ yd)", "Kısa (0-15 yd)"], rotation=0, va="center"
    )

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Passing Chart kaydedildi: {save_path}")


# 3. RUN GAP ANALİZİ (KORİDOR VERİMLİLİĞİ)
def plot_run_gap_analysis(df, team, save_path="run_gap_analysis.png"):
    """7 ana koşu koridorunun hacmini ve YPC (Yards Per Carry) değerini gösterir."""
    run_df = df[
        (df["posteam"] == team)
        & (df["play_type"] == "run")
        & (df["gap_detail"] != "Unknown")
    ].copy()

    # Soldan sağa saha koridor sırası
    ordered_gaps = [
        "L End",
        "L Tackle",
        "L Guard",
        "Center/A-Gap",
        "R Guard",
        "R Tackle",
        "R End",
    ]

    gap_summary = (
        run_df.groupby("gap_detail")
        .agg(
            rushes=("play_id", "count"),
            ypc=("yards_gained", "mean"),
            success_rate=("is_success", "mean"),
        )
        .reindex(ordered_gaps)
        .reset_index()
    )

    gap_summary["success_rate"] = gap_summary["success_rate"] * 100

    fig, ax1 = plt.subplots(figsize=(11, 5))
    ax2 = ax1.twinx()

    # Hacim (Bar Chart)
    x = np.arange(len(ordered_gaps))
    bars = ax1.bar(
        x,
        gap_summary["rushes"],
        width=0.45,
        color="#1f77b4",
        alpha=0.85,
        label="Koşu Sayısı",
    )

    # Verimlilik / YPC (Line Chart)
    line = ax2.plot(
        x,
        gap_summary["ypc"],
        color="#d62728",
        marker="o",
        linewidth=2.5,
        markersize=8,
        label="YPC (Yards Per Carry)",
    )

    # Etiketler ve Eksenler
    ax1.set_xlabel("Hücum Hattı Koridoru (Sol -> Sağ)", fontsize=11, labelpad=10)
    ax1.set_ylabel("Toplam Koşu Sayısı", fontsize=11, color="#1f77b4")
    ax2.set_ylabel("Ortalama Kazanılan Yard (YPC)", fontsize=11, color="#d62728")
    ax1.set_xticks(x)
    ax1.set_xticklabels(ordered_gaps, rotation=15)
    ax1.grid(axis="y", linestyle="--", alpha=0.5)
    ax2.grid(False)

    # Bar üstü değer etiketleri
    for bar in bars:
        h = bar.get_height()
        if not np.isnan(h) and h > 0:
            ax1.text(
                bar.get_x() + bar.get_width() / 2,
                h + 0.5,
                f"{int(h)}",
                ha="center",
                va="bottom",
                fontsize=9,
            )

    plt.title(
        f"{team} - Koşu Koridorları Tercih ve Verimlilik Analizi",
        fontsize=13,
        fontweight="bold",
        pad=15,
    )
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Run Gap grafiği kaydedildi: {save_path}")


# 4. DOWN & DISTANCE SUCCESS RATE MATRİSİ
def plot_success_rate_matrix(df, team, save_path="success_rate_matrix.png"):
    """Down ve mesafe kırılımında oyun tipine göre başarı yüzdesini çıkarır."""
    team_df = df[
        (df["posteam"] == team)
        & (df["play_type"].isin(["run", "pass"]))
        & (df["down"].isin([1, 2, 3, 4]))
    ].copy()

    def categorize_distance(dist):
        if dist <= 2:
            return "Short (1-2)"
        elif dist <= 6:
            return "Medium (3-6)"
        else:
            return "Long (7+)"

    team_df["dist_cat"] = team_df["ydstogo"].apply(categorize_distance)
    team_df["down_str"] = team_df["down"].astype(int).astype(str) + ". Down"

    pivot = (
        team_df.groupby(["down_str", "dist_cat", "play_type"])
        .agg(
            plays=("play_id", "count"),
            success_pct=("is_success", lambda x: np.mean(x) * 100),
        )
        .reset_index()
    )

    # Sadece yeterli örneklem olan durumlar (>= 3 oyun)
    pivot = pivot[pivot["plays"] >= 3]

    fig, ax = plt.subplots(figsize=(10, 5))
    sns.barplot(
        data=pivot,
        x="dist_cat",
        y="success_pct",
        hue="play_type",
        order=["Short (1-2)", "Medium (3-6)", "Long (7+)"],
        palette={"pass": "#3498db", "run": "#2ecc71"},
        ax=ax,
    )

    ax.axhline(50, color="gray", linestyle="--", linewidth=1, label="%50 Eşik")
    ax.set_title(
        f"{team} - Durumsal Başarı Oranı (Success Rate %)",
        fontsize=13,
        fontweight="bold",
        pad=15,
    )
    ax.set_xlabel("Mesafe Kategorisi", fontsize=11)
    ax.set_ylabel("Başarı Oranı (%)", fontsize=11)
    ax.set_ylim(0, 100)
    ax.legend(title="Oyun Tipi")

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Success Rate grafiği kaydedildi: {save_path}")


# --- TEST ÇALIŞTIRMASI ---
if __name__ == "__main__":
    print("Örnek veri çekiliyor (2023)...")
    cols = [
        "play_id",
        "play_type",
        "down",
        "ydstogo",
        "yards_gained",
        "posteam",
        "pass_length",
        "pass_location",
        "complete_pass",
        "run_location",
        "run_gap",
        "epa",
    ]
    pbp = nfl.import_pbp_data(
        [2023], columns=cols, downcast=True, include_participation=False
    )
    pbp_clean = enrich_pbp_data(pbp)

    # Örnek: San Francisco 49ers (SF)
    test_team = "SF"
    print(f"\n{test_team} için grafikler üretiliyor...")
    output_dir = Path(__file__).resolve().parents[1] / "output"
    output_dir.mkdir(parents=True, exist_ok=True)
    plot_passing_chart(pbp_clean, test_team, str(output_dir / "passing_chart.png"))
    plot_run_gap_analysis(
        pbp_clean, test_team, str(output_dir / "run_gap_analysis.png")
    )
    plot_success_rate_matrix(
        pbp_clean, test_team, str(output_dir / "success_rate_matrix.png")
    )