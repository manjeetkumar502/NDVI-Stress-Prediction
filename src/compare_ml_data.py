import pandas as pd

print("=" * 60)
print("VEGETATION AI - TRAINING vs REAL DATA CHECK")
print("=" * 60)

# Load old training dataset
old_df = pd.read_csv("data/processed/ml_dataset.csv")

# Load real Sentinel-2 dataset
real_df = pd.read_csv("data/processed/real_ml_features.csv")

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

print("\nOLD TRAINING DATA")
print("Shape:", old_df.shape)

print("\nREAL SENTINEL-2 DATA")
print("Shape:", real_df.shape)

print("\nFEATURE COMPARISON")
print("-" * 60)

for feature in features:

    old_min = old_df[feature].min()
    old_max = old_df[feature].max()
    old_mean = old_df[feature].mean()

    real_min = real_df[feature].min()
    real_max = real_df[feature].max()
    real_mean = real_df[feature].mean()

    print(f"\n{feature}")

    print(
        f"Old data  -> "
        f"min: {old_min:.6f}, "
        f"max: {old_max:.6f}, "
        f"mean: {old_mean:.6f}"
    )

    print(
        f"Real data -> "
        f"min: {real_min:.6f}, "
        f"max: {real_max:.6f}, "
        f"mean: {real_mean:.6f}"
    )

print("\n" + "=" * 60)
print("DATA COMPARISON COMPLETE")
print("=" * 60)