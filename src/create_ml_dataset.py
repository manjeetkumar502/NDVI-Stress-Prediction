import pandas as pd
import os

# ============================================================
# VEGETATION AI - CREATE ML DATASET
# ============================================================

DATA_PATH = "data/raw/sentinel2_sample.csv"
OUTPUT_PATH = "data/processed/ml_dataset.csv"

print("=" * 60)
print("VEGETATION AI - ML DATASET CREATION")
print("=" * 60)

# ------------------------------------------------------------
# 1. LOAD DATA
# ------------------------------------------------------------

df = pd.read_csv(DATA_PATH)

print("\nOriginal dataset:")
print(df.shape)

# ------------------------------------------------------------
# 2. CONVERT DATE
# ------------------------------------------------------------

df["date"] = pd.to_datetime(df["date"], errors="coerce")

# Sort chronologically within each field
df = df.sort_values(["field_id", "date"])

# ------------------------------------------------------------
# 3. CREATE FUTURE NDVI
# ------------------------------------------------------------

df["next_ndvi"] = (
    df.groupby("field_id")["NDVI"]
      .shift(-1)
)

# ------------------------------------------------------------
# 4. CALCULATE FUTURE NDVI CHANGE
# ------------------------------------------------------------

df["future_ndvi_change"] = (
    df["next_ndvi"] - df["NDVI"]
)

# ------------------------------------------------------------
# 5. REMOVE ROWS WITHOUT CURRENT/FUTURE NDVI
# ------------------------------------------------------------

df = df.dropna(
    subset=["NDVI", "next_ndvi"]
).copy()

print("\nRows after removing missing future NDVI:")
print(len(df))

# ------------------------------------------------------------
# 6. CREATE TARGET
# ------------------------------------------------------------

def classify_stress(change):

    if change > -0.05:
        return "Stable"

    elif change > -0.10:
        return "Moderate Stress"

    else:
        return "High Stress"


df["stress_class"] = df["future_ndvi_change"].apply(
    classify_stress
)

# ------------------------------------------------------------
# 7. CHECK TARGET DISTRIBUTION
# ------------------------------------------------------------

print("\nTARGET DISTRIBUTION")
print("-" * 40)

print(
    df["stress_class"].value_counts()
)

print("\nTARGET PERCENTAGE")
print("-" * 40)

print(
    df["stress_class"]
    .value_counts(normalize=True)
    .mul(100)
    .round(2)
)

# ------------------------------------------------------------
# 8. SELECT ML FEATURES
# ------------------------------------------------------------

features = [
    "NDVI",
    "EVI",
    "TCG",
    "B2",
    "B3",
    "B4",
    "B5",
    "B6",
    "B7",
    "B8",
    "B8A",
    "B9",
    "B11",
    "B12"
]

# Keep only required columns
ml_df = df[
    ["field_id", "date"] + features + ["stress_class"]
].copy()

# ------------------------------------------------------------
# 9. REMOVE MISSING FEATURE VALUES
# ------------------------------------------------------------

ml_df = ml_df.dropna(
    subset=features
).copy()

print("\nFinal ML dataset:")
print(ml_df.shape)

# ------------------------------------------------------------
# 10. SAVE DATASET
# ------------------------------------------------------------

os.makedirs(
    "data/processed",
    exist_ok=True
)

ml_df.to_csv(
    OUTPUT_PATH,
    index=False
)

print("\nSaved ML dataset to:")
print(OUTPUT_PATH)

print("\nFINAL COLUMNS")
print(ml_df.columns.tolist())

print("\n" + "=" * 60)
print("ML DATASET CREATION COMPLETE")
print("=" * 60)