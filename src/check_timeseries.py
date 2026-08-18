import pandas as pd

DATA_PATH = "data/raw/sentinel2_sample.csv"

print("=" * 60)
print("VEGETATION AI - TIME SERIES ANALYSIS")
print("=" * 60)

# Load dataset
df = pd.read_csv(DATA_PATH)

# Convert date
df["date"] = pd.to_datetime(df["date"], errors="coerce")

print("\n1. TOTAL ROWS")
print(len(df))

print("\n2. UNIQUE FIELDS")
print(df["field_id"].nunique())

print("\n3. DATE RANGE")
print(df["date"].min(), "to", df["date"].max())

print("\n4. UNIQUE DATES")
print(df["date"].nunique())

print("\n5. VALID NDVI OBSERVATIONS")
print(df["NDVI"].notna().sum())

print("\n6. AVERAGE OBSERVATIONS PER FIELD")
print(df.groupby("field_id").size().mean())

print("\n7. TOP 10 FIELDS BY OBSERVATION COUNT")
print(
    df.groupby("field_id")
      .size()
      .sort_values(ascending=False)
      .head(10)
)

print("\n8. OBSERVATIONS BY YEAR")
print(df["date"].dt.year.value_counts().sort_index())

print("\n" + "=" * 60)
print("TIME SERIES ANALYSIS COMPLETE")
print("=" * 60)