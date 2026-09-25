import pandas as pd

print("=" * 60)
print("VEGETATION AI - FINAL REAL DATA CLEANING")
print("=" * 60)

input_path = "data/processed/multiple_fields_timeseries.csv"
output_path = "data/processed/final_real_dataset.csv"

# ---------------------------------------------------------
# 1. LOAD DATA
# ---------------------------------------------------------

df = pd.read_csv(input_path)

df["date"] = pd.to_datetime(df["date"])

print("\nOriginal shape:")
print(df.shape)

# ---------------------------------------------------------
# 2. COUNT DUPLICATES
# ---------------------------------------------------------

duplicates = df.duplicated(
    subset=["field_id", "date"]
).sum()

print("\nDuplicate field/date records:")
print(duplicates)

# ---------------------------------------------------------
# 3. SORT BY CLOUD COVER
# ---------------------------------------------------------

df = df.sort_values(
    ["field_id", "date", "cloud_percentage"]
)

# ---------------------------------------------------------
# 4. KEEP LOWEST-CLOUD OBSERVATION
# ---------------------------------------------------------

df = df.drop_duplicates(
    subset=["field_id", "date"],
    keep="first"
)

print("\nShape after removing duplicates:")
print(df.shape)

# ---------------------------------------------------------
# 5. SORT FINAL DATA
# ---------------------------------------------------------

df = df.sort_values(
    ["field_id", "date"]
)

# ---------------------------------------------------------
# 6. SAVE
# ---------------------------------------------------------

df.to_csv(
    output_path,
    index=False
)

# ---------------------------------------------------------
# 7. FINAL SUMMARY
# ---------------------------------------------------------

print("\nFinal observations per field:")

print(
    df["field_id"]
    .value_counts()
    .sort_index()
)

print("\nRemaining duplicates:")

print(
    df.duplicated(
        subset=["field_id", "date"]
    ).sum()
)

print("\nFinal dataset shape:")

print(df.shape)

print("\nSaved to:")

print(output_path)

print("\n" + "=" * 60)
print("FINAL REAL DATA CLEANING COMPLETE")
print("=" * 60)