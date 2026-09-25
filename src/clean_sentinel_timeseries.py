import pandas as pd

# ---------------------------------------------------------
# VEGETATION AI - CLEAN REAL SENTINEL-2 MULTI-FIELD DATA
# ---------------------------------------------------------

input_path = "data/processed/multiple_fields_timeseries.csv"
output_path = "data/processed/final_real_dataset.csv"

print("=" * 60)
print("VEGETATION AI - MULTI-FIELD SENTINEL-2 DATA CLEANING")
print("=" * 60)


# ---------------------------------------------------------
# 1. LOAD DATA
# ---------------------------------------------------------

df = pd.read_csv(input_path)

df["date"] = pd.to_datetime(df["date"])

print("\nOriginal rows:", len(df))
print("Number of fields:", df["field_id"].nunique())


# ---------------------------------------------------------
# 2. CHECK OBSERVATIONS PER FIELD
# ---------------------------------------------------------

print("\nObservations per field:")

print(
    df.groupby("field_id").size().sort_index()
)


# ---------------------------------------------------------
# 3. REMOVE DUPLICATE FIELD + DATE RECORDS
# ---------------------------------------------------------

duplicates = df.duplicated(
    subset=["field_id", "date"]
).sum()

print("\nDuplicate field/date records:", duplicates)

df = df.sort_values(
    ["field_id", "date"]
)

df = df.drop_duplicates(
    subset=["field_id", "date"],
    keep="first"
)

print(
    "Rows after removing duplicates:",
    len(df)
)


# ---------------------------------------------------------
# 4. CHECK MISSING VALUES
# ---------------------------------------------------------

print("\nMissing values:")

missing = df.isnull().sum()

print(missing)


# ---------------------------------------------------------
# 5. REMOVE FIRST OBSERVATION OF EACH FIELD
# ---------------------------------------------------------

# Every field's first observation has no previous
# observation, so previous_ndvi, previous_evi,
# ndvi_change and evi_change are NaN.

before = len(df)

df = df.dropna(
    subset=[
        "previous_ndvi",
        "previous_evi",
        "ndvi_change",
        "evi_change"
    ]
)

removed = before - len(df)

print(
    "\nFirst observation removed from each field:",
    removed
)

print(
    "Rows after removing first observations:",
    len(df)
)


# ---------------------------------------------------------
# 6. CHECK NDVI RANGE
# ---------------------------------------------------------

print("\nNDVI range:")

print("Minimum:", df["NDVI"].min())
print("Maximum:", df["NDVI"].max())
print("Mean:", df["NDVI"].mean())


# ---------------------------------------------------------
# 7. CHECK EVI RANGE
# ---------------------------------------------------------

print("\nEVI range:")

print("Minimum:", df["EVI"].min())
print("Maximum:", df["EVI"].max())
print("Mean:", df["EVI"].mean())


# ---------------------------------------------------------
# 8. CHECK NDVI CHANGE
# ---------------------------------------------------------

print("\nNDVI change statistics:")

print(df["ndvi_change"].describe())


# ---------------------------------------------------------
# 9. FIND EXTREME NDVI CHANGES
# ---------------------------------------------------------

large_changes = df[
    df["ndvi_change"].abs() > 0.20
]

print(
    "\nLarge NDVI changes (absolute change > 0.20):",
    len(large_changes)
)

if len(large_changes) > 0:

    print("\nExtreme observations:")

    print(
        large_changes[
            [
                "field_id",
                "date",
                "NDVI",
                "previous_ndvi",
                "ndvi_change"
            ]
        ]
    )


# ---------------------------------------------------------
# 10. CHECK FINAL DUPLICATES
# ---------------------------------------------------------

final_duplicates = df.duplicated(
    subset=["field_id", "date"]
).sum()

print(
    "\nFinal duplicate field/date records:",
    final_duplicates
)


# ---------------------------------------------------------
# 11. CHECK FINAL MISSING VALUES
# ---------------------------------------------------------

final_missing = df.isnull().sum().sum()

print(
    "Final missing values:",
    final_missing
)


# ---------------------------------------------------------
# 12. FINAL FIELD DISTRIBUTION
# ---------------------------------------------------------

print("\nFinal observations per field:")

print(
    df.groupby("field_id").size().sort_index()
)


# ---------------------------------------------------------
# 13. SAVE CLEAN DATASET
# ---------------------------------------------------------

df.to_csv(
    output_path,
    index=False
)

print("\nClean dataset saved to:")
print(output_path)

print("\nFinal rows:", len(df))

print("\n" + "=" * 60)
print("MULTI-FIELD SENTINEL-2 DATA CLEANING COMPLETE")
print("=" * 60)