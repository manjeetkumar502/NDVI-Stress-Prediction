import pandas as pd
from sklearn.model_selection import train_test_split

# Load our processed ML dataset
df = pd.read_csv("data/processed/ml_dataset.csv")

print("Dataset shape:", df.shape)

# Features used by the model
features = [
    "NDVI",
    "EVI",
    "TCG",
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

# X = input features
X = df[features]

# y = target we want to predict
y = df["stress_class"]

print("\nFeatures:")
print(X.columns.tolist())

print("\nTarget:")
print(y.value_counts())

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nTraining data:", X_train.shape)
print("Testing data:", X_test.shape)
