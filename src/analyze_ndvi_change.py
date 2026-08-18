import pandas as pd

DATA_PATH = "data/raw/sentinel2_sample.csv"

print("=" * 60)
print("VEGETATION AI - NDVI CHANGE ANALYSIS")
print("=" * 60)

# Load data
df = pd.read_csv(DATA_PATH)

# Convert date
df["date"] = pd.to_datetime(df["date"], errors="coerce")

# Keep observations with NDVI
df = df.dropna(subset=["NDVI"])

# Sort by field and date
df = df.sort_values(["field_id", "date"])

# Previous NDVI for each field
df["previous_ndvi"] = df.groupby("field_id")["NDVI"].shift(1)

# NDVI change
df["ndvi_change"] = df["NDVI"] - df["previous_ndvi"]

# Remove first observation of each field
changes = df.dropna(subset=["ndvi_change"])

print("\n1. FIELDS")
print(changes["field_id"].nunique())

print("\n2. NDVI CHANGE OBSERVATIONS")
print(len(changes))

print("\n3. NDVI CHANGE STATISTICS")
print(changes["ndvi_change"].describe())

print("\n4. NDVI CHANGE QUANTILES")
print(
    changes["ndvi_change"].quantile(
        [0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99]
    )
)

print("\n5. SIGNIFICANT NDVI DECREASES")

for threshold in [-0.05, -0.10, -0.15, -0.20]:
    count = (changes["ndvi_change"] <= threshold).sum()
    percentage = count / len(changes) * 100

    print(
        f"Decrease <= {threshold}: "
        f"{count} observations ({percentage:.2f}%)"
    )

print("\n" + "=" * 60)
print("NDVI CHANGE ANALYSIS COMPLETE")
print("=" * 60)