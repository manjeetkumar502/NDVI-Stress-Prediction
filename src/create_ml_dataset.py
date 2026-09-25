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

df["date"] = pd.to_datetime(df["date"])

# Important:
# Sort each field according to time
df = df.sort_values(["field_id", "date"])


# ------------------------------------------------------------
# 3. GET PREVIOUS NDVI AND EVI
# ------------------------------------------------------------

df["previous_ndvi"] = (
    df.groupby("field_id")["NDVI"].shift(1)
)

df["previous_evi"] = (
    df.groupby("field_id")["EVI"].shift(1)
)


# ------------------------------------------------------------
# 4. CALCULATE NDVI AND EVI CHANGE
# ------------------------------------------------------------

df["ndvi_change"] = (
    df["NDVI"] - df["previous_ndvi"]
)

df["evi_change"] = (
    df["EVI"] - df["previous_evi"]
)


# ------------------------------------------------------------
# 5. GET NEXT NDVI
# ------------------------------------------------------------

df["next_ndvi"] = (
    df.groupby("field_id")["NDVI"].shift(-1)
)


# ------------------------------------------------------------
# 6. CALCULATE FUTURE NDVI CHANGE
# ------------------------------------------------------------

df["future_ndvi_change"] = (
    df["next_ndvi"] - df["NDVI"]
)


# ------------------------------------------------------------
# 7. REMOVE MISSING VALUES
# ------------------------------------------------------------

df = df.dropna(
    subset=[
        "NDVI",
        "EVI",
        "previous_ndvi",
        "previous_evi",
        "ndvi_change",
        "evi_change",
        "next_ndvi"
    ]
).copy()

print("\nRows after temporal processing:")
print(len(df))


# ------------------------------------------------------------
# 8. CREATE STRESS LABEL
# ------------------------------------------------------------

# We use future NDVI change to define vegetation condition.
#
# Large negative change = vegetation deterioration
# Small change          = moderate condition
# Positive change       = stable/improving vegetation
#
# Quantiles divide the observations into three groups.

low_threshold = df["future_ndvi_change"].quantile(0.33)
high_threshold = df["future_ndvi_change"].quantile(0.66)


def classify_stress(change):

    if change <= low_threshold:
        return "High Stress"

    elif change <= high_threshold:
        return "Moderate Stress"

    else:
        return "Stable"


df["stress_class"] = (
    df["future_ndvi_change"]
    .apply(classify_stress)
)


print("\nStress thresholds:")

print(
    "High Stress threshold:",
    round(low_threshold, 4)
)

print(
    "Stable threshold:",
    round(high_threshold, 4)
)

# ------------------------------------------------------------
# 9. SELECT FEATURES
# ------------------------------------------------------------

features = [
    "NDVI",
    "EVI",
    "TCG",

    "previous_ndvi",
    "previous_evi",

    "ndvi_change",
    "evi_change",

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


# ------------------------------------------------------------
# 10. CREATE FINAL ML DATASET
# ------------------------------------------------------------

ml_df = df[
    ["field_id", "date"] + features + ["stress_class"]
].copy()


# ------------------------------------------------------------
# 11. SAVE DATASET
# ------------------------------------------------------------

os.makedirs(
    "data/processed",
    exist_ok=True
)

ml_df.to_csv(
    OUTPUT_PATH,
    index=False
)


# ------------------------------------------------------------
# 12. PRINT RESULTS
# ------------------------------------------------------------

print("\nFinal ML dataset:")
print(ml_df.shape)

print("\nFeatures:")
print(features)

print("\nStress distribution:")
print(
    ml_df["stress_class"].value_counts()
)

print("\nSaved dataset to:")
print(OUTPUT_PATH)

print("\n" + "=" * 60)
print("ML DATASET CREATION COMPLETE")
print("=" * 60)
