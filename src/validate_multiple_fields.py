import pandas as pd

print("=" * 60)
print("VEGETATION AI - MULTI-FIELD DATA VALIDATION")
print("=" * 60)

# Load data
path = "data/processed/multiple_fields_timeseries.csv"

df = pd.read_csv(path)

print("\nDataset shape:")
print(df.shape)

# ---------------------------------------------------------
# 1. FIELDS
# ---------------------------------------------------------

print("\nNumber of fields:")
print(df["field_id"].nunique())

print("\nObservations per field:")
print(df["field_id"].value_counts().sort_index())

# ---------------------------------------------------------
# 2. DATE RANGE
# ---------------------------------------------------------

df["date"] = pd.to_datetime(df["date"])

print("\nDate range:")
print("Start:", df["date"].min())
print("End:", df["date"].max())

# ---------------------------------------------------------
# 3. DUPLICATES
# ---------------------------------------------------------

duplicates = df.duplicated(
    subset=["field_id", "date"]
).sum()

print("\nDuplicate field/date records:")
print(duplicates)

# ---------------------------------------------------------
# 4. MISSING VALUES
# ---------------------------------------------------------

print("\nMissing values:")

print(
    df.isnull().sum()
)

# ---------------------------------------------------------
# 5. NDVI
# ---------------------------------------------------------

print("\nNDVI statistics:")

print(
    df["NDVI"].describe()
)

# ---------------------------------------------------------
# 6. EVI
# ---------------------------------------------------------

print("\nEVI statistics:")

print(
    df["EVI"].describe()
)

# ---------------------------------------------------------
# 7. NDVI CHANGE
# ---------------------------------------------------------

print("\nNDVI change statistics:")

print(
    df["ndvi_change"].describe()
)

# ---------------------------------------------------------
# 8. CLOUD COVER
# ---------------------------------------------------------

print("\nCloud percentage statistics:")

print(
    df["cloud_percentage"].describe()
)

# ---------------------------------------------------------
# 9. EXTREME NDVI CHANGE
# ---------------------------------------------------------

extreme = df[
    df["ndvi_change"].abs() > 0.20
]

print(
    "\nExtreme NDVI changes (> 0.20):",
    len(extreme)
)

# ---------------------------------------------------------
# COMPLETE
# ---------------------------------------------------------

print("\n" + "=" * 60)
print("MULTI-FIELD VALIDATION COMPLETE")
print("=" * 60)