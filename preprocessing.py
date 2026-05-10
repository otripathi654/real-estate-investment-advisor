"""
=============================================================
 Real Estate Investment Advisor
 Step 1: Data Preprocessing + Feature Engineering
=============================================================
 Usage:
   python preprocessing.py --input data/india_housing_prices.csv
                            --output data/processed_data.csv

 What this script does:
   1. Loads raw CSV
   2. Cleans missing values & duplicates
   3. Engineers new features
   4. Creates both target variables
   5. Encodes categoricals & scales numericals
   6. Saves cleaned dataset ready for EDA + modelling
=============================================================
"""

import argparse
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
import warnings
import os

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
CURRENT_YEAR          = 2025
APPRECIATION_RATE     = 0.08        # 8% annual growth for future price
FUTURE_YEARS          = 5
GOOD_INV_SQFT_PCTILE  = 50          # price_per_sqft ≤ city median  → +1 point
GOOD_INV_AGE_THRESH   = 15          # Age ≤ 15 years                → +1 point
GOOD_INV_SCHOOL_THRESH= 3           # Nearby_Schools ≥ 3           → +1 point
GOOD_INV_SCORE_THRESH = 2           # total score ≥ 2 → Good Investment

NUMERICAL_COLS = [
    "BHK", "Size_in_SqFt", "Price_in_Lakhs", "Price_per_SqFt",
    "Floor_No", "Total_Floors", "Age_of_Property",
    "Nearby_Schools", "Nearby_Hospitals", "Parking_Space",
]

CATEGORICAL_COLS = [
    "State", "City", "Locality", "Property_Type",
    "Furnished_Status", "Facing", "Owner_Type",
    "Availability_Status", "Security", "Amenities",
]


# ─────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────

def load_data(path: str) -> pd.DataFrame:
    """Load CSV and do a quick audit."""
    print(f"\n{'='*55}")
    print(f"  Loading dataset: {path}")
    print(f"{'='*55}")
    df = pd.read_csv(path)
    print(f"  Shape           : {df.shape[0]:,} rows × {df.shape[1]} columns")
    print(f"  Columns         : {list(df.columns)}")
    print(f"  Missing values  :\n{df.isnull().sum()[df.isnull().sum() > 0]}")
    print(f"  Duplicate rows  : {df.duplicated().sum():,}")
    return df


def remove_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """Drop exact duplicate rows."""
    before = len(df)
    df = df.drop_duplicates()
    print(f"\n[Duplicates] Removed {before - len(df):,} rows → {len(df):,} remain")
    return df.reset_index(drop=True)


def impute_missing(df: pd.DataFrame) -> pd.DataFrame:
    """
    Numerical  → median imputation (robust to outliers)
    Categorical → mode imputation
    """
    print("\n[Imputation] Filling missing values …")

    for col in NUMERICAL_COLS:
        if col in df.columns and df[col].isnull().sum() > 0:
            median_val = df[col].median()
            df[col].fillna(median_val, inplace=True)
            print(f"  {col:30s} → median = {median_val:.2f}")

    for col in CATEGORICAL_COLS:
        if col in df.columns and df[col].isnull().sum() > 0:
            mode_val = df[col].mode()[0]
            df[col].fillna(mode_val, inplace=True)
            print(f"  {col:30s} → mode  = {mode_val}")

    remaining = df.isnull().sum().sum()
    print(f"  Remaining nulls after imputation: {remaining}")
    return df


def remove_outliers(df: pd.DataFrame) -> pd.DataFrame:
    """
    IQR-based capping (Winsorisation) on key price/size columns.
    We cap rather than drop to preserve dataset size.
    """
    print("\n[Outliers] IQR-based capping on Price_in_Lakhs & Size_in_SqFt …")
    for col in ["Price_in_Lakhs", "Size_in_SqFt", "Price_per_SqFt"]:
        if col not in df.columns:
            continue
        Q1, Q3 = df[col].quantile(0.25), df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower, upper = Q1 - 1.5 * IQR, Q3 + 1.5 * IQR
        before_outliers = ((df[col] < lower) | (df[col] > upper)).sum()
        df[col] = df[col].clip(lower, upper)
        print(f"  {col:30s} clipped {before_outliers:,} values  [{lower:.1f}, {upper:.1f}]")
    return df


# ─────────────────────────────────────────────
# FEATURE ENGINEERING
# ─────────────────────────────────────────────

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create all derived features required by the project brief."""
    print("\n[Feature Engineering] Creating new features …")

    # 1. Age of Property (if not already present)
    if "Year_Built" in df.columns:
        df["Age_of_Property"] = CURRENT_YEAR - df["Year_Built"]
        df["Age_of_Property"] = df["Age_of_Property"].clip(lower=0)
        print("  ✔ Age_of_Property")

    # 2. Price per SqFt (recalculate / fill if missing)
    if "Size_in_SqFt" in df.columns and "Price_in_Lakhs" in df.columns:
        df["Price_per_SqFt"] = (
            df["Price_in_Lakhs"] * 100_000 / df["Size_in_SqFt"].replace(0, np.nan)
        ).round(2)
        df["Price_per_SqFt"].fillna(df["Price_per_SqFt"].median(), inplace=True)
        print("  ✔ Price_per_SqFt (recalculated)")

    # 3. Floor Ratio  (how high up is the property?)
    if "Floor_No" in df.columns and "Total_Floors" in df.columns:
        df["Floor_Ratio"] = (
            df["Floor_No"] / df["Total_Floors"].replace(0, np.nan)
        ).fillna(0).round(3)
        print("  ✔ Floor_Ratio")

    # 4. School Density Score  (normalised 0-1)
    if "Nearby_Schools" in df.columns:
        max_schools = df["Nearby_Schools"].max()
        df["School_Density_Score"] = (
            df["Nearby_Schools"] / max_schools if max_schools > 0 else 0
        ).round(3)
        print("  ✔ School_Density_Score")

    # 5. Hospital Density Score
    if "Nearby_Hospitals" in df.columns:
        max_hosp = df["Nearby_Hospitals"].max()
        df["Hospital_Density_Score"] = (
            df["Nearby_Hospitals"] / max_hosp if max_hosp > 0 else 0
        ).round(3)
        print("  ✔ Hospital_Density_Score")

    # 6. Infrastructure Score  (composite of schools + hospitals + transport)
    transport_map = {"Low": 1, "Medium": 2, "High": 3}
    if "Public_Transport_Accessibility" in df.columns:
        df["Transport_Score"] = (
            df["Public_Transport_Accessibility"].map(transport_map).fillna(1)
        )
        df["Infrastructure_Score"] = (
            df["School_Density_Score"]
            + df["Hospital_Density_Score"]
            + df["Transport_Score"] / 3
        ).round(3)
        print("  ✔ Infrastructure_Score")

    # 7. Total Amenity Count  (comma-separated string → count)
    if "Amenities" in df.columns:
        df["Amenity_Count"] = (
            df["Amenities"]
            .astype(str)
            .apply(lambda x: len([a for a in x.split(",") if a.strip() != "" and a.strip().lower() != "nan"]))
        )
        print("  ✔ Amenity_Count")

    # 8. Is New Property  (built in last 5 years)
    if "Age_of_Property" in df.columns:
        df["Is_New_Property"] = (df["Age_of_Property"] <= 5).astype(int)
        print("  ✔ Is_New_Property")

    # 9. Has Premium Security
    if "Security" in df.columns:
        df["Has_Premium_Security"] = (
            df["Security"].astype(str).str.contains("Gated|CCTV|Guard", case=False, na=False)
        ).astype(int)
        print("  ✔ Has_Premium_Security")

    # 10. Is Fully Furnished
    if "Furnished_Status" in df.columns:
        df["Is_Fully_Furnished"] = (
            df["Furnished_Status"].str.strip().str.lower() == "fully"
        ).astype(int)
        print("  ✔ Is_Fully_Furnished")

    return df


# ─────────────────────────────────────────────
# TARGET VARIABLE CREATION
# ─────────────────────────────────────────────

def create_target_variables(df: pd.DataFrame) -> pd.DataFrame:
    """
    Target 1 — Good_Investment  (Classification, binary 0/1)
        Multi-factor scoring:
          +1  if Price_per_SqFt ≤ city median
          +1  if Age_of_Property ≤ 15 years
          +1  if Nearby_Schools  ≥ 3
        Score ≥ 2 → Good Investment (1), else 0

    Target 2 — Future_Price_5yr  (Regression, in Lakhs)
        Future = Current_Price × (1 + 0.08)^5
        With a small location adjustment bonus for high-infrastructure cities.
    """
    print("\n[Targets] Creating target variables …")

    # ── Classification target ──────────────────
    score = pd.Series(0, index=df.index)

    if "Price_per_SqFt" in df.columns and "City" in df.columns:
        city_median = df.groupby("City")["Price_per_SqFt"].transform("median")
        score += (df["Price_per_SqFt"] <= city_median).astype(int)

    if "Age_of_Property" in df.columns:
        score += (df["Age_of_Property"] <= GOOD_INV_AGE_THRESH).astype(int)

    if "Nearby_Schools" in df.columns:
        score += (df["Nearby_Schools"] >= GOOD_INV_SCHOOL_THRESH).astype(int)

    df["Good_Investment"] = (score >= GOOD_INV_SCORE_THRESH).astype(int)
    dist = df["Good_Investment"].value_counts()
    print(f"  ✔ Good_Investment  →  0: {dist.get(0,0):,}  |  1: {dist.get(1,0):,}  "
          f"(balance: {dist.get(1,0)/len(df)*100:.1f}% positive)")

    # ── Regression target ──────────────────────
    if "Price_in_Lakhs" in df.columns:
        # Base compound growth
        base_multiplier = (1 + APPRECIATION_RATE) ** FUTURE_YEARS   # ≈ 1.469

        # Small location bonus: cities with above-average infrastructure
        location_bonus = pd.Series(0.0, index=df.index)
        if "Infrastructure_Score" in df.columns:
            city_infra = df.groupby("City")["Infrastructure_Score"].transform("mean") \
                if "City" in df.columns else df["Infrastructure_Score"]
            overall_mean = city_infra.mean()
            location_bonus = (city_infra > overall_mean).astype(float) * 0.02  # +2% bonus

        df["Future_Price_5yr"] = (
            df["Price_in_Lakhs"] * (base_multiplier + location_bonus)
        ).round(2)

        print(f"  ✔ Future_Price_5yr  →  mean ≈ ₹{df['Future_Price_5yr'].mean():.1f}L  "
              f"| min ≈ ₹{df['Future_Price_5yr'].min():.1f}L  "
              f"| max ≈ ₹{df['Future_Price_5yr'].max():.1f}L")

    return df


# ─────────────────────────────────────────────
# ENCODING & SCALING
# ─────────────────────────────────────────────

def encode_and_scale(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """
    Encode categoricals with Label Encoding.
    Scale selected numericals with MinMaxScaler.
    Returns (transformed df, dict of fitted encoders/scalers for reuse in app).
    """
    print("\n[Encoding & Scaling] …")
    artifacts = {}

    # Label encode high-cardinality cats (State, City, Locality)
    high_card = ["State", "City", "Locality"]
    for col in high_card:
        if col in df.columns:
            le = LabelEncoder()
            df[f"{col}_enc"] = le.fit_transform(df[col].astype(str))
            artifacts[f"le_{col}"] = le
            print(f"  ✔ LabelEncoded  {col} → {col}_enc  ({df[col].nunique()} categories)")

    # One-hot encode low-cardinality cats
    low_card = [
        "Property_Type", "Furnished_Status", "Facing",
        "Owner_Type", "Availability_Status",
    ]
    for col in low_card:
        if col in df.columns:
            dummies = pd.get_dummies(df[col].astype(str), prefix=col, drop_first=False)
            df = pd.concat([df, dummies], axis=1)
            print(f"  ✔ OneHotEncoded {col}  ({df[col].nunique()} cats → {dummies.shape[1]} cols)")

    # MinMax scale core numericals
    scale_cols = [
        c for c in [
            "Size_in_SqFt", "Price_per_SqFt", "Age_of_Property",
            "Nearby_Schools", "Nearby_Hospitals", "Parking_Space",
            "Floor_Ratio", "School_Density_Score", "Hospital_Density_Score",
            "Infrastructure_Score", "Amenity_Count",
        ] if c in df.columns
    ]
    scaler = MinMaxScaler()
    
    df = df.replace({"Yes": 1, "No": 0})

    df[scale_cols] = scaler.fit_transform(df[scale_cols])
    artifacts["scaler"] = scaler
    artifacts["scale_cols"] = scale_cols
    print(f"  ✔ MinMaxScaled  {len(scale_cols)} columns")

    return df, artifacts


# ─────────────────────────────────────────────
# SAVE
# ─────────────────────────────────────────────

def save_outputs(df: pd.DataFrame, artifacts: dict, output_path: str):
    """Save processed CSV and encoding artifacts."""
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)

    df.to_csv(output_path, index=False)
    print(f"\n[Saved] Processed dataset → {output_path}  ({df.shape[0]:,} rows × {df.shape[1]} cols)")

    # Save column list for reference
    col_path = output_path.replace(".csv", "_columns.txt")
    with open(col_path, "w") as f:
        f.write("\n".join(df.columns.tolist()))
    print(f"[Saved] Column list        → {col_path}")

    # Save quick summary
    summary_path = output_path.replace(".csv", "_summary.txt")
    with open(summary_path, "w") as f:
        f.write("=== Processed Dataset Summary ===\n\n")
        f.write(f"Shape : {df.shape}\n\n")
        f.write("Dtypes:\n")
        f.write(df.dtypes.to_string())
        f.write("\n\nDescriptive Stats:\n")
        f.write(df.describe().to_string())
    print(f"[Saved] Summary stats      → {summary_path}")


# ─────────────────────────────────────────────
# MAIN PIPELINE
# ─────────────────────────────────────────────

def run_pipeline(input_path: str, output_path: str):
    df = load_data(input_path)
    df = remove_duplicates(df)
    df = impute_missing(df)
    df = remove_outliers(df)
    df = engineer_features(df)
    df = create_target_variables(df)
    df, artifacts = encode_and_scale(df)
    save_outputs(df, artifacts, output_path)

    print("\n" + "="*55)
    print("  ✅  Preprocessing complete!")
    print(f"  Final shape : {df.shape}")
    print(f"  Targets     : Good_Investment, Future_Price_5yr")
    print("="*55 + "\n")

    return df, artifacts


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Real Estate — Data Preprocessing")
    parser.add_argument("--input",  default="data/india_housing_prices.csv",
                        help="Path to raw CSV")
    parser.add_argument("--output", default="data/processed_data.csv",
                        help="Path to save processed CSV")
    args = parser.parse_args()
    run_pipeline(args.input, args.output)
