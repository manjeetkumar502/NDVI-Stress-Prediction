import pandas as pd

DATA_PATH = "data/raw/sentinel2_sample.csv"

print("=" * 60)
print("VEGETATION AI - DATASET INSPECTION")
print("=" * 60)

df = pd.read_csv(DATA_PATH)

print("\n1. DATASET SHAPE")
print(df.shape)

print("\n2. COLUMNS")
print(df.columns.tolist())

print("\n3. FIRST 5 ROWS")
print(df.head())

print("\n4. DATA TYPES")
print(df.dtypes)

print("\n5. MISSING VALUES")
print(df.isnull().sum())

print("\n6. DUPLICATE ROWS")
print(df.duplicated().sum())

print("\n7. NUMERICAL STATISTICS")
print(df.describe())

print("\n8. NDVI STATISTICS")
print(df["NDVI"].describe())

print("\n9. VALID COUNTS")
print(df["valid"].value_counts(dropna=False))

print("\n" + "=" * 60)
print("INSPECTION COMPLETE")
print("=" * 60)