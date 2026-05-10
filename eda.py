"""
=============================================================
 Real Estate Investment Advisor
 Step 2: Exploratory Data Analysis (EDA) — All 20 Questions
=============================================================
 Usage:
   python eda.py --input data/processed_data.csv
                 --output_dir eda_outputs/

 Sections:
   Q01–Q05  : Price & Size Analysis
   Q06–Q10  : Location-based Analysis
   Q11–Q15  : Feature Relationship & Correlation
   Q16–Q20  : Investment / Amenities / Ownership Analysis

 Each question saves a PNG chart to output_dir.
 A final HTML report index is also written.
=============================================================
"""

import argparse
import os
from os import path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
import warnings

warnings.filterwarnings("ignore")

# ── Global style ──────────────────────────────────────────
sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams.update({
    "figure.dpi": 130,
    "figure.facecolor": "white",
    "axes.titlesize": 14,
    "axes.titleweight": "bold",
    "axes.labelsize": 11,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
})
ACCENT   = "#2563EB"   # blue
POSITIVE = "#16A34A"   # green
NEGATIVE = "#DC2626"   # red
PALETTE  = "Blues_d"


def save(fig, path):
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✔  Saved → {path}")


def load(input_path):
    df = pd.read_csv(input_path)
    print(f"Loaded: {df.shape[0]:,} rows × {df.shape[1]} cols")

    # Ensure raw price column exists (unscaled); eda works best on originals
    # If the user runs eda on processed_data.csv, Price_in_Lakhs may be scaled.
    # We keep it as-is and label axes accordingly.
    return df


# ═══════════════════════════════════════════════════════════
#  SECTION 1 — Price & Size Analysis  (Q01–Q05)
# ═══════════════════════════════════════════════════════════

def q01_price_distribution(df, out):
    """Q1: What is the distribution of property prices?"""
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    col = "Price_in_Lakhs"
    if col not in df.columns:
        print(f"  ⚠  Column '{col}' not found, skipping Q01"); return

    axes[0].hist(df[col], bins=50, color=ACCENT, edgecolor="white", alpha=0.85)
    axes[0].set_title("Distribution of Property Prices")
    axes[0].set_xlabel("Price (Lakhs)")
    axes[0].set_ylabel("Count")

    axes[1].boxplot(df[col].dropna(), vert=False, patch_artist=True,
                    boxprops=dict(facecolor=ACCENT, alpha=0.6),
                    medianprops=dict(color=NEGATIVE, linewidth=2))
    axes[1].set_title("Price — Box Plot (outlier view)")
    axes[1].set_xlabel("Price (Lakhs)")

    fig.suptitle("Q1 · Property Price Distribution", fontsize=15, fontweight="bold")
    save(fig, os.path.join(out, "Q01_price_distribution.png"))


def q02_size_distribution(df, out):
    """Q2: What is the distribution of property sizes?"""
    col = "Size_in_SqFt"
    if col not in df.columns:
        print(f"  ⚠  Column '{col}' not found, skipping Q02"); return

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    axes[0].hist(df[col], bins=50, color="#7C3AED", edgecolor="white", alpha=0.85)
    axes[0].set_title("Distribution of Property Sizes")
    axes[0].set_xlabel("Size (SqFt)")
    axes[0].set_ylabel("Count")

    axes[1].boxplot(df[col].dropna(), vert=False, patch_artist=True,
                    boxprops=dict(facecolor="#7C3AED", alpha=0.6),
                    medianprops=dict(color=NEGATIVE, linewidth=2))
    axes[1].set_title("Size — Box Plot")
    axes[1].set_xlabel("Size (SqFt)")

    fig.suptitle("Q2 · Property Size Distribution", fontsize=15, fontweight="bold")
    save(fig, os.path.join(out, "Q02_size_distribution.png"))


def q03_price_per_sqft_by_type(df, out):
    """Q3: How does price per sq ft vary by property type?"""
    col_x, col_y = "Property_Type", "Price_per_SqFt"
    # try encoded column
    if col_x not in df.columns and col_y not in df.columns:
        print(f"  ⚠  Columns needed for Q03 not found, skipping"); return

    fig, ax = plt.subplots(figsize=(12, 6))
    order = (df.groupby(col_x)[col_y].median().sort_values(ascending=False).index
             if col_x in df.columns else None)

    if col_x in df.columns:
        sns.boxplot(data=df, x=col_x, y=col_y, order=order,
                    palette="Set2", ax=ax, fliersize=2)
        ax.set_xlabel("Property Type")
    else:
        ax.text(0.5, 0.5, "Property_Type column not in processed data\n(check encoding step)",
                ha="center", va="center", transform=ax.transAxes)

    ax.set_title("Q3 · Price per SqFt by Property Type", fontsize=15, fontweight="bold")
    ax.set_ylabel("Price per SqFt")
    ax.tick_params(axis="x", rotation=20)
    save(fig, os.path.join(out, "Q03_price_per_sqft_by_type.png"))


def q04_size_vs_price(df, out):
    """Q4: Is there a relationship between property size and price?"""
    cx, cy = "Size_in_SqFt", "Price_in_Lakhs"
    if cx not in df.columns or cy not in df.columns:
        print(f"  ⚠  Columns needed for Q04 not found, skipping"); return

    fig, ax = plt.subplots(figsize=(10, 6))
    sample = df.sample(min(3000, len(df)), random_state=42)
    ax.scatter(sample[cx], sample[cy], alpha=0.35, s=18, color=ACCENT)

    # Regression line
    m, b = np.polyfit(sample[cx].fillna(0), sample[cy].fillna(0), 1)
    xs = np.linspace(sample[cx].min(), sample[cx].max(), 200)
    ax.plot(xs, m * xs + b, color=NEGATIVE, linewidth=2, label=f"Trend (slope={m:.4f})")

    corr = sample[cx].corr(sample[cy])
    ax.set_title(f"Q4 · Size vs Price  (Pearson r = {corr:.3f})", fontsize=15, fontweight="bold")
    ax.set_xlabel("Size (SqFt)")
    ax.set_ylabel("Price (Lakhs)")
    ax.legend()
    save(fig, os.path.join(out, "Q04_size_vs_price.png"))


def q05_outliers_price_sqft(df, out):
    """Q5: Are there any outliers in price per sq ft or property size?"""
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    targets = [("Price_per_SqFt", axes[0], "#F59E0B"),
               ("Size_in_SqFt",   axes[1], "#10B981")]

    for col, ax, color in targets:
        if col not in df.columns:
            ax.text(0.5, 0.5, f"{col} not found", ha="center", va="center",
                    transform=ax.transAxes)
            continue
        Q1, Q3 = df[col].quantile(0.25), df[col].quantile(0.75)
        IQR = Q3 - Q1
        outliers = df[(df[col] < Q1 - 1.5*IQR) | (df[col] > Q3 + 1.5*IQR)]
        ax.boxplot(df[col].dropna(), vert=True, patch_artist=True,
                   boxprops=dict(facecolor=color, alpha=0.6),
                   medianprops=dict(color=NEGATIVE, linewidth=2),
                   flierprops=dict(marker="o", markersize=3, alpha=0.4, color=NEGATIVE))
        ax.set_title(f"{col}\n({len(outliers):,} outliers detected)")
        ax.set_ylabel(col)

    fig.suptitle("Q5 · Outlier Detection", fontsize=15, fontweight="bold")
    save(fig, os.path.join(out, "Q05_outliers.png"))


# ═══════════════════════════════════════════════════════════
#  SECTION 2 — Location-based Analysis  (Q06–Q10)
# ═══════════════════════════════════════════════════════════

def q06_avg_price_per_sqft_by_state(df, out):
    """Q6: What is the average price per sq ft by state?"""
    if "State" not in df.columns or "Price_per_SqFt" not in df.columns:
        print("  ⚠  Skipping Q06"); return

    agg = df.groupby("State")["Price_per_SqFt"].mean().sort_values(ascending=True)
    fig, ax = plt.subplots(figsize=(12, max(6, len(agg) * 0.35)))
    bars = ax.barh(agg.index, agg.values, color=ACCENT, alpha=0.8, edgecolor="white")
    ax.bar_label(bars, fmt="%.0f", padding=3, fontsize=8)
    ax.set_title("Q6 · Avg Price per SqFt by State", fontsize=15, fontweight="bold")
    ax.set_xlabel("Avg Price per SqFt")
    save(fig, os.path.join(out, "Q06_avg_price_by_state.png"))


def q07_avg_price_by_city(df, out):
    """Q7: What is the average property price by city?"""
    if "City" not in df.columns or "Price_in_Lakhs" not in df.columns:
        print("  ⚠  Skipping Q07"); return

    agg = (df.groupby("City")["Price_in_Lakhs"].mean()
             .sort_values(ascending=False).head(20))
    fig, ax = plt.subplots(figsize=(13, 6))
    sns.barplot(x=agg.index, y=agg.values, palette="Blues_r", ax=ax)
    ax.set_title("Q7 · Top 20 Cities by Avg Property Price", fontsize=15, fontweight="bold")
    ax.set_xlabel("City")
    ax.set_ylabel("Avg Price (Lakhs)")
    ax.tick_params(axis="x", rotation=45)
    save(fig, os.path.join(out, "Q07_avg_price_by_city.png"))


def q08_median_age_by_locality(df, out):
    """Q8: What is the median age of properties by locality?"""
    if "Locality" not in df.columns or "Age_of_Property" not in df.columns:
        print("  ⚠  Skipping Q08"); return

    agg = (df.groupby("Locality")["Age_of_Property"].median()
             .sort_values(ascending=False).head(20))
    fig, ax = plt.subplots(figsize=(13, 6))
    sns.barplot(x=agg.index, y=agg.values, palette="YlOrRd_r", ax=ax)
    ax.set_title("Q8 · Top 20 Localities by Median Property Age", fontsize=15, fontweight="bold")
    ax.set_xlabel("Locality")
    ax.set_ylabel("Median Age (Years)")
    ax.tick_params(axis="x", rotation=45)
    save(fig, os.path.join(out, "Q08_median_age_by_locality.png"))


def q09_bhk_distribution_by_city(df, out):
    """Q9: How is BHK distributed across cities?"""
    if "City" not in df.columns or "BHK" not in df.columns:
        print("  ⚠  Skipping Q09"); return

    top_cities = df["City"].value_counts().head(8).index
    sub = df[df["City"].isin(top_cities)]
    fig, ax = plt.subplots(figsize=(13, 6))
    ct = pd.crosstab(sub["City"], sub["BHK"])
    ct.plot(kind="bar", ax=ax, colormap="tab10", edgecolor="white", width=0.75)
    ax.set_title("Q9 · BHK Distribution Across Top 8 Cities", fontsize=15, fontweight="bold")
    ax.set_xlabel("City")
    ax.set_ylabel("Count")
    ax.tick_params(axis="x", rotation=30)
    ax.legend(title="BHK", bbox_to_anchor=(1.01, 1))
    save(fig, os.path.join(out, "Q09_bhk_by_city.png"))


def q10_top5_expensive_localities(df, out):
    """Q10: What are the price trends for the top 5 most expensive localities?"""
    if "Locality" not in df.columns or "Price_in_Lakhs" not in df.columns:
        print("  ⚠  Skipping Q10"); return

    top5 = (df.groupby("Locality")["Price_in_Lakhs"].mean()
              .sort_values(ascending=False).head(5).index)
    sub = df[df["Locality"].isin(top5)]

    fig, ax = plt.subplots(figsize=(12, 6))
    for loc in top5:
        d = sub[sub["Locality"] == loc]["Price_in_Lakhs"].sort_values()
        ax.plot(range(len(d)), d.values, marker="o", markersize=3,
                linewidth=1.5, alpha=0.8, label=loc)
    ax.set_title("Q10 · Price Trends — Top 5 Most Expensive Localities",
                 fontsize=15, fontweight="bold")
    ax.set_xlabel("Property Index (sorted by price)")
    ax.set_ylabel("Price (Lakhs)")
    ax.legend(bbox_to_anchor=(1.01, 1))
    save(fig, os.path.join(out, "Q10_top5_locality_trends.png"))


# ═══════════════════════════════════════════════════════════
#  SECTION 3 — Feature Relationships & Correlation  (Q11–Q15)
# ═══════════════════════════════════════════════════════════

def q11_correlation_heatmap(df, out):
    """Q11: How are numeric features correlated with each other?"""
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    # exclude target and encoded cols for cleaner heatmap
    exclude = [c for c in num_cols if "_enc" in c or c.startswith("Property_Type_")
               or c.startswith("Furnished") or c.startswith("Facing")
               or c.startswith("Owner") or c.startswith("Avail")]
    num_cols = [c for c in num_cols if c not in exclude][:20]

    corr = df[num_cols].corr()
    fig, ax = plt.subplots(figsize=(14, 10))
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap="coolwarm",
                center=0, linewidths=0.5, ax=ax,
                annot_kws={"size": 8}, cbar_kws={"shrink": 0.8})
    ax.set_title("Q11 · Numeric Feature Correlation Heatmap", fontsize=15, fontweight="bold")
    save(fig, os.path.join(out, "Q11_correlation_heatmap.png"))


def q12_schools_vs_price(df, out):
    """Q12: How do nearby schools relate to price per sq ft?"""
    cx, cy = "Nearby_Schools", "Price_per_SqFt"
    if cx not in df.columns or cy not in df.columns:
        print("  ⚠  Skipping Q12"); return

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    sample = df.sample(min(3000, len(df)), random_state=42)
    axes[0].scatter(sample[cx], sample[cy], alpha=0.3, s=15, color="#0EA5E9")
    axes[0].set_xlabel("Nearby Schools")
    axes[0].set_ylabel("Price per SqFt")
    axes[0].set_title("Scatter: Schools vs Price/SqFt")

    agg = df.groupby(df[cx].round(1))[cy].mean()
    axes[1].bar(agg.index, agg.values, color="#0EA5E9", alpha=0.8, width=0.08)
    axes[1].set_xlabel("Nearby Schools (rounded)")
    axes[1].set_ylabel("Avg Price per SqFt")
    axes[1].set_title("Avg Price per SqFt by School Count")

    fig.suptitle("Q12 · Nearby Schools → Price per SqFt", fontsize=15, fontweight="bold")
    save(fig, os.path.join(out, "Q12_schools_vs_price.png"))


def q13_hospitals_vs_price(df, out):
    """Q13: How do nearby hospitals relate to price per sq ft?"""
    cx, cy = "Nearby_Hospitals", "Price_per_SqFt"
    if cx not in df.columns or cy not in df.columns:
        print("  ⚠  Skipping Q13"); return

    fig, ax = plt.subplots(figsize=(10, 5))
    agg = df.groupby(df[cx].round(1))[cy].mean().reset_index()
    ax.plot(agg[cx], agg[cy], marker="o", color="#8B5CF6", linewidth=2)
    ax.fill_between(agg[cx], agg[cy], alpha=0.15, color="#8B5CF6")
    ax.set_title("Q13 · Nearby Hospitals → Avg Price per SqFt", fontsize=15, fontweight="bold")
    ax.set_xlabel("Nearby Hospitals")
    ax.set_ylabel("Avg Price per SqFt")
    save(fig, os.path.join(out, "Q13_hospitals_vs_price.png"))


def q14_price_by_furnished_status(df, out):
    """Q14: How does price vary by furnished status?"""
    col = "Furnished_Status"
    cy  = "Price_in_Lakhs"
    if col not in df.columns or cy not in df.columns:
        print("  ⚠  Skipping Q14"); return

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    order = df.groupby(col)[cy].median().sort_values(ascending=False).index

    sns.boxplot(data=df, x=col, y=cy, order=order, palette="Set2", ax=axes[0], fliersize=2)
    axes[0].set_title("Price Distribution by Furnished Status")
    axes[0].set_xlabel("Furnished Status")
    axes[0].set_ylabel("Price (Lakhs)")

    agg = df.groupby(col)[cy].mean().reindex(order)
    axes[1].bar(agg.index, agg.values, color=sns.color_palette("Set2", len(agg)), edgecolor="white")
    axes[1].set_title("Avg Price by Furnished Status")
    axes[1].set_xlabel("Furnished Status")
    axes[1].set_ylabel("Avg Price (Lakhs)")

    fig.suptitle("Q14 · Price vs Furnished Status", fontsize=15, fontweight="bold")
    save(fig, os.path.join(out, "Q14_price_by_furnished.png"))


def q15_price_by_facing(df, out):
    """Q15: How does price per sq ft vary by property facing direction?"""
    col = "Facing"
    cy  = "Price_per_SqFt"
    if col not in df.columns or cy not in df.columns:
        print("  ⚠  Skipping Q15"); return

    agg = df.groupby(col)[cy].mean().sort_values(ascending=False)
    fig, ax = plt.subplots(figsize=(10, 5))
    colors = sns.color_palette("husl", len(agg))
    bars = ax.bar(agg.index, agg.values, color=colors, edgecolor="white")
    ax.bar_label(bars, fmt="%.0f", padding=3, fontsize=9)
    ax.set_title("Q15 · Avg Price per SqFt by Facing Direction", fontsize=15, fontweight="bold")
    ax.set_xlabel("Facing Direction")
    ax.set_ylabel("Avg Price per SqFt")
    ax.tick_params(axis="x", rotation=20)
    save(fig, os.path.join(out, "Q15_price_by_facing.png"))


# ═══════════════════════════════════════════════════════════
#  SECTION 4 — Investment / Amenities / Ownership  (Q16–Q20)
# ═══════════════════════════════════════════════════════════

def q16_owner_type_distribution(df, out):
    """Q16: How many properties belong to each owner type?"""
    col = "Owner_Type"
    if col not in df.columns:
        print("  ⚠  Skipping Q16"); return

    counts = df[col].value_counts()
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].pie(counts.values, labels=counts.index, autopct="%1.1f%%",
                colors=sns.color_palette("pastel"), startangle=140,
                wedgeprops=dict(edgecolor="white"))
    axes[0].set_title("Owner Type — Pie Chart")

    sns.barplot(x=counts.index, y=counts.values, palette="pastel", ax=axes[1])
    axes[1].set_title("Owner Type — Bar Chart")
    axes[1].set_xlabel("Owner Type")
    axes[1].set_ylabel("Count")

    fig.suptitle("Q16 · Property Distribution by Owner Type", fontsize=15, fontweight="bold")
    save(fig, os.path.join(out, "Q16_owner_type_distribution.png"))


def q17_availability_status(df, out):
    """Q17: How many properties are available under each availability status?"""
    col = "Availability_Status"
    if col not in df.columns:
        print("  ⚠  Skipping Q17"); return

    counts = df[col].value_counts()
    fig, ax = plt.subplots(figsize=(9, 5))
    colors = [POSITIVE, ACCENT, NEGATIVE][:len(counts)]
    bars = ax.bar(counts.index, counts.values, color=colors, edgecolor="white", width=0.5)
    ax.bar_label(bars, fmt="%d", padding=3, fontsize=10)
    ax.set_title("Q17 · Properties by Availability Status", fontsize=15, fontweight="bold")
    ax.set_xlabel("Availability Status")
    ax.set_ylabel("Count")
    save(fig, os.path.join(out, "Q17_availability_status.png"))


def q18_parking_vs_price(df, out):
    """Q18: Does parking space affect property price?"""
    cx, cy = "Parking_Space", "Price_in_Lakhs"
    if cx not in df.columns or cy not in df.columns:
        print("  ⚠  Skipping Q18"); return

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    sns.boxplot(data=df, x=df[cx].round(0).astype(int), y=cy,
                palette="Blues", ax=axes[0], fliersize=2)
    axes[0].set_title("Price Distribution by Parking Spaces")
    axes[0].set_xlabel("Parking Spaces")
    axes[0].set_ylabel("Price (Lakhs)")

    agg = df.groupby(df[cx].round(0).astype(int))[cy].mean()
    axes[1].plot(agg.index, agg.values, marker="s", color=ACCENT, linewidth=2)
    axes[1].set_title("Avg Price vs Parking Count")
    axes[1].set_xlabel("Parking Spaces")
    axes[1].set_ylabel("Avg Price (Lakhs)")

    fig.suptitle("Q18 · Parking Space → Property Price", fontsize=15, fontweight="bold")
    save(fig, os.path.join(out, "Q18_parking_vs_price.png"))


def q19_amenities_vs_price(df, out):
    """Q19: How do amenities affect price per sq ft?"""
    cx, cy = "Amenity_Count", "Price_per_SqFt"
    # fallback: Amenities raw column
    if cx not in df.columns and "Amenities" in df.columns:
        df[cx] = df["Amenities"].astype(str).apply(
            lambda x: len([a for a in x.split(",") if a.strip() not in ("", "nan")])
        )

    if cx not in df.columns or cy not in df.columns:
        print("  ⚠  Skipping Q19"); return

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    sns.boxplot(data=df, x=df[cx].round(0).astype(int), y=cy,
                palette="Greens", ax=axes[0], fliersize=2)
    axes[0].set_title("Price/SqFt by Amenity Count")
    axes[0].set_xlabel("Number of Amenities")
    axes[0].set_ylabel("Price per SqFt")

    agg = df.groupby(df[cx].round(0).astype(int))[cy].mean()
    bars = axes[1].bar(agg.index, agg.values,
                       color=sns.color_palette("Greens_d", len(agg)), edgecolor="white")
    axes[1].bar_label(bars, fmt="%.0f", padding=2, fontsize=8)
    axes[1].set_title("Avg Price/SqFt by Amenity Count")
    axes[1].set_xlabel("Number of Amenities")
    axes[1].set_ylabel("Avg Price per SqFt")

    fig.suptitle("Q19 · Amenities → Price per SqFt", fontsize=15, fontweight="bold")
    save(fig, os.path.join(out, "Q19_amenities_vs_price.png"))


def q20_transport_vs_investment(df, out):
    """Q20: How does public transport accessibility relate to price/sqft or investment potential?"""
    col_t = "Public_Transport_Accessibility"
    col_p = "Price_per_SqFt"
    col_g = "Good_Investment"

    if col_t not in df.columns:
        print("  ⚠  Skipping Q20"); return

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    if col_p in df.columns:
        agg = df.groupby(col_t)[col_p].mean().sort_values(ascending=False)
        colors_t = [ACCENT, "#60A5FA", "#BFDBFE"][:len(agg)]
        bars = axes[0].bar(agg.index, agg.values, color=colors_t, edgecolor="white")
        axes[0].bar_label(bars, fmt="%.0f", padding=3)
        axes[0].set_title("Avg Price/SqFt by Transport Access")
        axes[0].set_xlabel("Transport Accessibility")
        axes[0].set_ylabel("Avg Price per SqFt")

    if col_g in df.columns:
        ct = pd.crosstab(df[col_t], df[col_g], normalize="index") * 100
        ct.columns = ["Not Good Inv.", "Good Investment"]
        ct.plot(kind="bar", ax=axes[1], color=[NEGATIVE, POSITIVE],
                edgecolor="white", width=0.6)
        axes[1].set_title("Good Investment % by Transport Access")
        axes[1].set_xlabel("Transport Accessibility")
        axes[1].set_ylabel("% of Properties")
        axes[1].tick_params(axis="x", rotation=0)
        axes[1].legend(loc="upper right")
    else:
        axes[1].text(0.5, 0.5, "Good_Investment target not found\n(run preprocessing first)",
                     ha="center", va="center", transform=axes[1].transAxes)

    fig.suptitle("Q20 · Public Transport → Price & Investment Potential",
                 fontsize=15, fontweight="bold")
    save(fig, os.path.join(out, "Q20_transport_vs_investment.png"))


# ═══════════════════════════════════════════════════════════
#  HTML REPORT INDEX
# ═══════════════════════════════════════════════════════════

QUESTIONS = [
    ("Q01", "Price & Size", "Distribution of property prices"),
    ("Q02", "Price & Size", "Distribution of property sizes"),
    ("Q03", "Price & Size", "Price per SqFt by property type"),
    ("Q04", "Price & Size", "Relationship between size and price"),
    ("Q05", "Price & Size", "Outliers in price per SqFt and size"),
    ("Q06", "Location",     "Avg price per SqFt by state"),
    ("Q07", "Location",     "Avg property price by city"),
    ("Q08", "Location",     "Median age of properties by locality"),
    ("Q09", "Location",     "BHK distribution across cities"),
    ("Q10", "Location",     "Price trends — top 5 expensive localities"),
    ("Q11", "Correlation",  "Numeric feature correlation heatmap"),
    ("Q12", "Correlation",  "Nearby schools → price per SqFt"),
    ("Q13", "Correlation",  "Nearby hospitals → price per SqFt"),
    ("Q14", "Correlation",  "Price by furnished status"),
    ("Q15", "Correlation",  "Price per SqFt by facing direction"),
    ("Q16", "Investment",   "Properties by owner type"),
    ("Q17", "Investment",   "Properties by availability status"),
    ("Q18", "Investment",   "Parking space → property price"),
    ("Q19", "Investment",   "Amenities → price per SqFt"),
    ("Q20", "Investment",   "Public transport → price & investment"),
]

SECTION_COLORS = {
    "Price & Size": "#2563EB",
    "Location":     "#7C3AED",
    "Correlation":  "#0F766E",
    "Investment":   "#B45309",
}

def write_html_report(out_dir):
    imgs = ""
    for qid, section, desc in QUESTIONS:
        filename = next((f for f in os.listdir(out_dir) if f.startswith(qid)), None)
        if not filename:
            continue
        color = SECTION_COLORS.get(section, "#374151")
        imgs += f"""
        <div class="card">
          <span class="badge" style="background:{color}">{section}</span>
          <h3>{qid} · {desc}</h3>
          <img src="{filename}" alt="{qid}">
        </div>"""

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <title>Real Estate EDA Report</title>
  <style>
    body {{ font-family: system-ui, sans-serif; background:#F9FAFB; color:#111827; margin:0; padding:24px; }}
    h1   {{ text-align:center; color:#1E3A5F; margin-bottom:4px; }}
    p.sub{{ text-align:center; color:#6B7280; margin-top:0; margin-bottom:32px; }}
    .grid{{ display:grid; grid-template-columns: repeat(auto-fill, minmax(520px,1fr)); gap:24px; }}
    .card{{ background:#fff; border-radius:12px; box-shadow:0 2px 8px rgba(0,0,0,.08);
            padding:20px; }}
    .card h3 {{ margin:8px 0 12px; font-size:15px; }}
    .card img {{ width:100%; border-radius:8px; }}
    .badge {{ display:inline-block; font-size:11px; font-weight:600; color:#fff;
              padding:2px 10px; border-radius:99px; margin-bottom:4px; }}
  </style>
</head>
<body>
  <h1>🏠 Real Estate Investment Advisor — EDA Report</h1>
  <p class="sub">20 Exploratory Data Analysis questions • india_housing_prices dataset</p>
  <div class="grid">{imgs}</div>
</body>
</html>"""

    path = os.path.join(out_dir, "eda_report.html")
    with open(path, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"\n  ✔  HTML report → {path}")


# ═══════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════

def run_eda(input_path: str, output_dir: str):
    os.makedirs(output_dir, exist_ok=True)
    df = load(input_path)

    print("\n── Section 1: Price & Size Analysis ─────────────────")
    q01_price_distribution(df, output_dir)
    q02_size_distribution(df, output_dir)
    q03_price_per_sqft_by_type(df, output_dir)
    q04_size_vs_price(df, output_dir)
    q05_outliers_price_sqft(df, output_dir)

    print("\n── Section 2: Location-based Analysis ───────────────")
    q06_avg_price_per_sqft_by_state(df, output_dir)
    q07_avg_price_by_city(df, output_dir)
    q08_median_age_by_locality(df, output_dir)
    q09_bhk_distribution_by_city(df, output_dir)
    q10_top5_expensive_localities(df, output_dir)

    print("\n── Section 3: Feature Relationships & Correlation ───")
    q11_correlation_heatmap(df, output_dir)
    q12_schools_vs_price(df, output_dir)
    q13_hospitals_vs_price(df, output_dir)
    q14_price_by_furnished_status(df, output_dir)
    q15_price_by_facing(df, output_dir)

    print("\n── Section 4: Investment / Amenities / Ownership ────")
    q16_owner_type_distribution(df, output_dir)
    q17_availability_status(df, output_dir)
    q18_parking_vs_price(df, output_dir)
    q19_amenities_vs_price(df, output_dir)
    q20_transport_vs_investment(df, output_dir)

    write_html_report(output_dir)

    print(f"\n{'='*55}")
    print("  ✅  EDA complete! All 20 charts saved.")
    print(f"  📁  Output folder : {output_dir}")
    print(f"  🌐  Open report   : {output_dir}/eda_report.html")
    print(f"{'='*55}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Real Estate — EDA")
    parser.add_argument("--input",      default="data/processed_data.csv")
    parser.add_argument("--output_dir", default="eda_outputs/")
    args = parser.parse_args()
    run_eda(args.input, args.output_dir)
