"""
Analysis: Does height, weight, or gender affect CrossFit Open placement?

Outputs:
  - crossfit_data/analysis_summary.txt   (text report)
  - crossfit_data/plots/                 (visualizations)
"""

import pandas as pd
import numpy as np
from scipy import stats
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

INPUT = Path("crossfit_data/crossfit_open_leaderboard.csv")
PLOT_DIR = Path("crossfit_data/plots")
PLOT_DIR.mkdir(exist_ok=True)

# ---------------------------------------------------------------------------
# Load & clean
# ---------------------------------------------------------------------------
print("Loading data...")
df = pd.read_csv(INPUT, low_memory=False)
print(f"Raw rows: {len(df):,}")

# Convert types
df["overall_rank"] = pd.to_numeric(df["overall_rank"], errors="coerce")
df["height_inches"] = pd.to_numeric(df["height_inches"], errors="coerce")
df["weight_lbs"] = pd.to_numeric(df["weight_lbs"], errors="coerce")
df["age"] = pd.to_numeric(df["age"], errors="coerce")
df["year"] = pd.to_numeric(df["year"], errors="coerce")

for i in range(1, 7):
    df[f"wod{i}_rank"] = pd.to_numeric(df[f"wod{i}_rank"], errors="coerce")

# Filter to rows with physical data
has_height = df["height_inches"].notna() & (df["height_inches"] > 48) & (df["height_inches"] < 96)
has_weight = df["weight_lbs"].notna() & (df["weight_lbs"] > 80) & (df["weight_lbs"] < 400)
has_rank = df["overall_rank"].notna()

df_phys = df[has_height & has_weight & has_rank].copy()
print(f"Rows with valid height+weight+rank: {len(df_phys):,} ({len(df_phys)/len(df)*100:.1f}%)")

# Percentile rank (0=best, 100=worst) within each year+division
df_phys["pct_rank"] = df_phys.groupby(["year", "division"])["overall_rank"].transform(
    lambda x: x.rank(pct=True) * 100
)

# Also compute per-workout percentile ranks
for i in range(1, 7):
    col = f"wod{i}_rank"
    df_phys[f"wod{i}_pct"] = df_phys.groupby(["year", "division"])[col].transform(
        lambda x: x.rank(pct=True) * 100
    )

# BMI
df_phys["bmi"] = (df_phys["weight_lbs"] * 703) / (df_phys["height_inches"] ** 2)

# Tier buckets
df_phys["tier"] = pd.cut(
    df_phys["pct_rank"],
    bins=[0, 1, 5, 10, 25, 50, 100],
    labels=["Top 1%", "Top 5%", "Top 10%", "Top 25%", "Top 50%", "Bottom 50%"],
)

report = []
report.append("=" * 70)
report.append("CROSSFIT OPEN ANALYSIS: HEIGHT, WEIGHT & GENDER vs PLACEMENT")
report.append("=" * 70)

# ---------------------------------------------------------------------------
# 1. Data coverage
# ---------------------------------------------------------------------------
report.append("\n## DATA COVERAGE\n")
coverage = df.groupby(["year", "division"]).agg(
    total=("overall_rank", "size"),
    has_height=("height_inches", lambda x: x.notna().sum()),
    has_weight=("weight_lbs", lambda x: x.notna().sum()),
).reset_index()
coverage["height_pct"] = (coverage["has_height"] / coverage["total"] * 100).round(1)
coverage["weight_pct"] = (coverage["has_weight"] / coverage["total"] * 100).round(1)
report.append(coverage.to_string(index=False))

# ---------------------------------------------------------------------------
# 2. Correlation: height/weight vs overall percentile rank
# ---------------------------------------------------------------------------
report.append("\n\n## CORRELATIONS: PHYSICAL ATTRIBUTES vs OVERALL PERCENTILE RANK")
report.append("(negative r = taller/heavier athletes rank BETTER)\n")

for div in ["Men", "Women"]:
    report.append(f"\n### {div}")
    sub = df_phys[df_phys["division"] == div]

    for attr, label in [("height_inches", "Height"), ("weight_lbs", "Weight"), ("bmi", "BMI")]:
        corrs = []
        for year in sorted(sub["year"].unique()):
            ys = sub[sub["year"] == year]
            if len(ys) < 100:
                continue
            r, p = stats.spearmanr(ys[attr], ys["pct_rank"])
            corrs.append({"year": year, "r": round(r, 4), "p": f"{p:.2e}", "n": len(ys)})

        cdf = pd.DataFrame(corrs)
        report.append(f"\n  {label} vs Percentile Rank (Spearman):")
        if len(cdf) > 0:
            report.append(f"  Mean r across years: {cdf['r'].mean():.4f}")
            report.append(cdf.to_string(index=False))
        else:
            report.append("  Insufficient data")

# ---------------------------------------------------------------------------
# 3. Physical profiles by performance tier
# ---------------------------------------------------------------------------
report.append("\n\n## PHYSICAL PROFILE BY PERFORMANCE TIER")

for div in ["Men", "Women"]:
    report.append(f"\n### {div}")
    sub = df_phys[df_phys["division"] == div]
    tier_stats = sub.groupby("tier", observed=True).agg(
        n=("height_inches", "size"),
        avg_height=("height_inches", "mean"),
        avg_weight=("weight_lbs", "mean"),
        avg_bmi=("bmi", "mean"),
        avg_age=("age", "mean"),
    ).round(1)
    report.append(tier_stats.to_string())

# ---------------------------------------------------------------------------
# 4. Per-workout correlations (which workouts favor size?)
# ---------------------------------------------------------------------------
report.append("\n\n## PER-WORKOUT CORRELATIONS WITH HEIGHT & WEIGHT")
report.append("(Shows which specific workouts favor taller/heavier athletes)\n")

wod_corrs = []
for div in ["Men", "Women"]:
    sub = df_phys[df_phys["division"] == div]
    for year in sorted(sub["year"].unique()):
        ys = sub[sub["year"] == year]
        for wod in range(1, 7):
            pct_col = f"wod{wod}_pct"
            if pct_col not in ys.columns or ys[pct_col].notna().sum() < 100:
                continue
            valid = ys[[pct_col, "height_inches", "weight_lbs"]].dropna()
            if len(valid) < 100:
                continue
            rh, _ = stats.spearmanr(valid["height_inches"], valid[pct_col])
            rw, _ = stats.spearmanr(valid["weight_lbs"], valid[pct_col])
            wod_corrs.append({
                "division": div, "year": year, "workout": f"{year}.{wod}",
                "r_height": round(rh, 4), "r_weight": round(rw, 4), "n": len(valid),
            })

wod_df = pd.DataFrame(wod_corrs)
if len(wod_df) > 0:
    # Show workouts most affected by size (sorted by absolute weight correlation)
    for div in ["Men", "Women"]:
        report.append(f"\n### {div} — Workouts most influenced by weight:")
        dsub = wod_df[wod_df["division"] == div].copy()
        dsub["abs_r_weight"] = dsub["r_weight"].abs()
        top = dsub.nlargest(15, "abs_r_weight")[["workout", "r_height", "r_weight", "n"]]
        report.append(top.to_string(index=False))

        report.append(f"\n### {div} — Workouts most influenced by height:")
        dsub["abs_r_height"] = dsub["r_height"].abs()
        top = dsub.nlargest(15, "abs_r_height")[["workout", "r_height", "r_weight", "n"]]
        report.append(top.to_string(index=False))

# ---------------------------------------------------------------------------
# 5. Optimal body type by year
# ---------------------------------------------------------------------------
report.append("\n\n## OPTIMAL BODY TYPE (Top 1% vs Field Average)")

for div in ["Men", "Women"]:
    report.append(f"\n### {div}")
    sub = df_phys[df_phys["division"] == div]
    rows = []
    for year in sorted(sub["year"].unique()):
        ys = sub[sub["year"] == year]
        top1 = ys[ys["pct_rank"] <= 1]
        if len(top1) < 10:
            continue
        rows.append({
            "year": year,
            "top1_height": round(top1["height_inches"].mean(), 1),
            "field_height": round(ys["height_inches"].mean(), 1),
            "top1_weight": round(top1["weight_lbs"].mean(), 1),
            "field_weight": round(ys["weight_lbs"].mean(), 1),
            "top1_bmi": round(top1["bmi"].mean(), 1),
            "field_bmi": round(ys["bmi"].mean(), 1),
            "top1_age": round(top1["age"].mean(), 1),
            "field_age": round(ys["age"].mean(), 1),
            "n_top1": len(top1),
        })
    if rows:
        report.append(pd.DataFrame(rows).to_string(index=False))

# ---------------------------------------------------------------------------
# 6. Statistical tests
# ---------------------------------------------------------------------------
report.append("\n\n## STATISTICAL TESTS")
report.append("Mann-Whitney U: Top 10% vs Bottom 50% physical attributes\n")

for div in ["Men", "Women"]:
    report.append(f"\n### {div}")
    sub = df_phys[df_phys["division"] == div]
    top10 = sub[sub["pct_rank"] <= 10]
    bottom50 = sub[sub["pct_rank"] > 50]

    for attr, label in [("height_inches", "Height"), ("weight_lbs", "Weight"), ("bmi", "BMI")]:
        a = top10[attr].dropna()
        b = bottom50[attr].dropna()
        if len(a) > 30 and len(b) > 30:
            u_stat, p_val = stats.mannwhitneyu(a, b, alternative="two-sided")
            effect = a.mean() - b.mean()
            report.append(f"  {label}: Top10% mean={a.mean():.1f}, Bottom50% mean={b.mean():.1f}, "
                          f"diff={effect:+.1f}, p={p_val:.2e}")

# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------
print("Generating plots...")

sns.set_theme(style="whitegrid", font_scale=1.1)

# Plot 1: Height/Weight distributions by tier
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
for col, (div, axrow) in enumerate(zip(["Men", "Women"], axes)):
    sub = df_phys[df_phys["division"] == div]
    sns.boxplot(data=sub, x="tier", y="height_inches", ax=axrow[0], palette="RdYlGn_r")
    axrow[0].set_title(f"{div} — Height by Performance Tier")
    axrow[0].set_xlabel("")
    axrow[0].set_ylabel("Height (inches)")
    axrow[0].tick_params(axis="x", rotation=30)

    sns.boxplot(data=sub, x="tier", y="weight_lbs", ax=axrow[1], palette="RdYlGn_r")
    axrow[1].set_title(f"{div} — Weight by Performance Tier")
    axrow[1].set_xlabel("")
    axrow[1].set_ylabel("Weight (lbs)")
    axrow[1].tick_params(axis="x", rotation=30)

plt.tight_layout()
plt.savefig(PLOT_DIR / "tier_distributions.png", dpi=150)
plt.close()

# Plot 2: Correlation heatmap — weight vs rank by year
fig, axes = plt.subplots(1, 2, figsize=(16, 6))
for ax, div in zip(axes, ["Men", "Women"]):
    dsub = wod_df[wod_df["division"] == div].copy()
    if len(dsub) == 0:
        continue
    pivot = dsub.pivot_table(index="year", columns="workout", values="r_weight")
    # Only keep columns that match the year
    for year in pivot.index:
        for col in pivot.columns:
            if not col.startswith(str(year)):
                pivot.loc[year, col] = np.nan
    # Reshape: year x workout number
    records = []
    for _, row in dsub.iterrows():
        wod_num = int(row["workout"].split(".")[1])
        records.append({"year": row["year"], "wod": f"WOD {wod_num}", "r": row["r_weight"]})
    hdf = pd.DataFrame(records)
    hpivot = hdf.pivot_table(index="year", columns="wod", values="r")
    sns.heatmap(hpivot, annot=True, fmt=".3f", cmap="RdBu_r", center=0, ax=ax,
                vmin=-0.15, vmax=0.15)
    ax.set_title(f"{div} — Weight vs Workout Rank Correlation")

plt.tight_layout()
plt.savefig(PLOT_DIR / "weight_workout_heatmap.png", dpi=150)
plt.close()

# Plot 3: Top 1% vs field average over time
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
for col, div in enumerate(["Men", "Women"]):
    sub = df_phys[df_phys["division"] == div]
    yearly = []
    for year in sorted(sub["year"].unique()):
        ys = sub[sub["year"] == year]
        top1 = ys[ys["pct_rank"] <= 1]
        if len(top1) < 10:
            continue
        yearly.append({
            "year": year,
            "top1_height": top1["height_inches"].mean(),
            "field_height": ys["height_inches"].mean(),
            "top1_weight": top1["weight_lbs"].mean(),
            "field_weight": ys["weight_lbs"].mean(),
        })
    ydf = pd.DataFrame(yearly)
    if len(ydf) == 0:
        continue

    ax = axes[0][col]
    ax.plot(ydf["year"], ydf["top1_height"], "o-", label="Top 1%", color="red")
    ax.plot(ydf["year"], ydf["field_height"], "s--", label="Field avg", color="gray")
    ax.set_title(f"{div} — Height Over Time")
    ax.set_ylabel("Height (inches)")
    ax.legend()

    ax = axes[1][col]
    ax.plot(ydf["year"], ydf["top1_weight"], "o-", label="Top 1%", color="red")
    ax.plot(ydf["year"], ydf["field_weight"], "s--", label="Field avg", color="gray")
    ax.set_title(f"{div} — Weight Over Time")
    ax.set_ylabel("Weight (lbs)")
    ax.legend()

plt.tight_layout()
plt.savefig(PLOT_DIR / "top1_vs_field_trend.png", dpi=150)
plt.close()

# Plot 4: Scatter — height vs weight colored by percentile rank
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
for ax, div in zip(axes, ["Men", "Women"]):
    sub = df_phys[df_phys["division"] == div].sample(n=min(20000, len(df_phys)), random_state=42)
    scatter = ax.scatter(
        sub["height_inches"], sub["weight_lbs"],
        c=sub["pct_rank"], cmap="RdYlGn", alpha=0.3, s=5, vmin=0, vmax=100,
    )
    ax.set_title(f"{div} — Height vs Weight (color=percentile rank)")
    ax.set_xlabel("Height (inches)")
    ax.set_ylabel("Weight (lbs)")
    plt.colorbar(scatter, ax=ax, label="Percentile Rank (0=best)")

plt.tight_layout()
plt.savefig(PLOT_DIR / "height_weight_scatter.png", dpi=150)
plt.close()

# ---------------------------------------------------------------------------
# Save report
# ---------------------------------------------------------------------------
report_text = "\n".join(report)
print(report_text)

with open("crossfit_data/analysis_summary.txt", "w") as f:
    f.write(report_text)

print(f"\nReport saved: crossfit_data/analysis_summary.txt")
print(f"Plots saved: {PLOT_DIR}/")
