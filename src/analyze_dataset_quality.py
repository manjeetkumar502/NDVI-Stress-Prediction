import pandas as pd

DATA_PATH = "data/raw/sentinel2_sample.csv"

df = pd.read_csv(DATA_PATH)
df["date"] = pd.to_datetime(df["date"])

print("=" * 60)
print("VEGETATION AI - DATASET QUALITY ANALYSIS")
print("=" * 60)

print("\n1. DATASET")
print("Rows:", len(df))
print("Fields:", df["field_id"].nunique())

print("\n2. TIME COVERAGE")
print("Start date:", df["date"].min())
print("End date:", df["date"].max())
print("Unique dates:", df["date"].nunique())

print("\n3. OBSERVATIONS PER DATE")
print(df.groupby("date").size())

print("\n4. VALID NDVI OBSERVATIONS")
print(df["NDVI"].notna().sum())

print("\n5. NDVI RANGE")
print("Minimum:", df["NDVI"].min())
print("Maximum:", df["NDVI"].max())
print("Mean:", df["NDVI"].mean())

print("\n6. OBSERVATIONS PER FIELD")

field_counts = df.groupby("field_id")["NDVI"].count()

print("Average:", field_counts.mean())
print("Minimum:", field_counts.min())
print("Maximum:", field_counts.max())

print("\nFields with at least 10 observations:",
      (field_counts >= 10).sum())

print("Fields with at least 5 observations:",
      (field_counts >= 5).sum())

print("\n" + "=" * 60)
print("DATASET QUALITY ANALYSIS COMPLETE")
print("=" * 60)