import pandas as pd

# ---------------------------------------------------------
# VEGETATION AI - REAL DATA ML FEATURE ENGINEERING
# ---------------------------------------------------------

input_path = "data/processed/final_real_dataset.csv"
output_path = "data/processed/real_ml_features.csv"

print("=" * 60)
print("VEGETATION AI - REAL ML FEATURE ENGINEERING")
print("=" * 60)


# ---------------------------------------------------------
# 1. LOAD FINAL REAL SATELLITE DATA
# ---------------------------------------------------------

df = pd.read_csv(input_path)

df["date"] = pd.to_datetime(df["date"])

print("\nInput dataset shape:")
print(df.shape)

print("\nNumber of fields:")
print(df["field_id"].nunique())


# ---------------------------------------------------------
# 2. SORT DATA
# ---------------------------------------------------------

df = df.sort_values(
    ["field_id", "date"]
).reset_index(drop=True)


# ---------------------------------------------------------
# 3. ML FEATURES
# ---------------------------------------------------------

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


# ---------------------------------------------------------
# 4. CHECK REQUIRED FEATURES
# ---------------------------------------------------------

missing_features = [
    feature
    for feature in features
    if feature not in df.columns
]

if missing_features:
    print("\nERROR: Missing required features:")
    for feature in missing_features:
        print("-", feature)

    raise ValueError(
        "Required ML features are missing from the dataset."
    )


# ---------------------------------------------------------
# 5. CREATE ML DATASET
# ---------------------------------------------------------

ml_df = df[
    ["field_id", "date"] + features
].copy()


# ---------------------------------------------------------
# 6. CHECK MISSING VALUES
# ---------------------------------------------------------

print("\nMissing values:")

missing_count = ml_df[features].isnull().sum().sum()

print(missing_count)


# ---------------------------------------------------------
# 7. REMOVE ROWS WITH MISSING ML FEATURES
# ---------------------------------------------------------

before_rows = len(ml_df)

ml_df = ml_df.dropna(
    subset=features
).reset_index(drop=True)

after_rows = len(ml_df)

print("\nRows removed because of missing ML features:")
print(before_rows - after_rows)


# ---------------------------------------------------------
# 8. CHECK FEATURE RANGES
# ---------------------------------------------------------

print("\nFeature sanity check:")

print(
    "\nNDVI range:",
    round(ml_df["NDVI"].min(), 4),
    "to",
    round(ml_df["NDVI"].max(), 4)
)

print(
    "EVI range:",
    round(ml_df["EVI"].min(), 4),
    "to",
    round(ml_df["EVI"].max(), 4)
)

print(
    "NDVI change range:",
    round(ml_df["ndvi_change"].min(), 4),
    "to",
    round(ml_df["ndvi_change"].max(), 4)
)

print(
    "EVI change range:",
    round(ml_df["evi_change"].min(), 4),
    "to",
    round(ml_df["evi_change"].max(), 4)
)


# ---------------------------------------------------------
# 9. DISPLAY FEATURES
# ---------------------------------------------------------

print("\nML Features:")

for feature in features:
    print("-", feature)


# ---------------------------------------------------------
# 10. FIELD-WISE DATA CHECK
# ---------------------------------------------------------

print("\nObservations per field:")

print(
    ml_df.groupby("field_id").size()
)


# ---------------------------------------------------------
# 11. FINAL DATASET INFORMATION
# ---------------------------------------------------------

print("\nFinal ML feature dataset:")
print(ml_df.shape)

print("\nColumns:")
print(list(ml_df.columns))


# ---------------------------------------------------------
# 12. SAVE
# ---------------------------------------------------------

ml_df.to_csv(
    output_path,
    index=False
)

print("\nSaved to:")
print(output_path)

print("\n" + "=" * 60)
print("REAL ML FEATURE ENGINEERING COMPLETE")
print("=" * 60)