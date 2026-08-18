from datasets import load_dataset
import pandas as pd
import os

print("Loading Sentinel-2 dataset...")

dataset = load_dataset(
    "jaelin215/sentinel-2-reanalysis-NDVI-EVI-TCG",
    split="train",
    streaming=True
)

print("Dataset connected successfully.")

# Take a manageable sample
sample_size = 100000

rows = []

for i, row in enumerate(dataset):
    rows.append(row)

    if (i + 1) % 10000 == 0:
        print(f"Collected {i + 1} rows...")

    if i + 1 >= sample_size:
        break

df = pd.DataFrame(rows)

print("\nDataset shape:", df.shape)
print("\nColumns:")
print(df.columns.tolist())

# Create output directory
os.makedirs("data/raw", exist_ok=True)

output_path = "data/raw/sentinel2_sample.csv"

df.to_csv(output_path, index=False)

print(f"\nSaved dataset to: {output_path}")
print("DATA DOWNLOAD COMPLETE!")