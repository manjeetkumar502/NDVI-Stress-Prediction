import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import accuracy_score, classification_report


# --------------------------------------------------
# 1. LOAD DATA
# --------------------------------------------------

df = pd.read_csv("data/processed/ml_dataset.csv")

print("Dataset shape:", df.shape)


# --------------------------------------------------
# 2. SELECT FEATURES
# --------------------------------------------------

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

X = df[features]
y = df["stress_class"]


# --------------------------------------------------
# 3. SPLIT DATA
# --------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nTraining data:", X_train.shape)
print("Testing data:", X_test.shape)


# ==================================================
# 4. LOGISTIC REGRESSION
# ==================================================

logistic_model = Pipeline([
    ("scaler", StandardScaler()),

    ("classifier", LogisticRegression(
        max_iter=1000,
        class_weight="balanced"
    ))
])

print("\nTraining Logistic Regression...")

logistic_model.fit(X_train, y_train)

logistic_pred = logistic_model.predict(X_test)

logistic_accuracy = accuracy_score(
    y_test,
    logistic_pred
)

print("Logistic Regression Accuracy:",
      round(logistic_accuracy, 4))

print("\nLogistic Regression Report:")
print(classification_report(
    y_test,
    logistic_pred
))


# ==================================================
# 5. RANDOM FOREST
# ==================================================

random_forest_model = RandomForestClassifier(
    n_estimators=200,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1
)

print("\nTraining Random Forest...")

random_forest_model.fit(X_train, y_train)

random_forest_pred = random_forest_model.predict(X_test)

random_forest_accuracy = accuracy_score(
    y_test,
    random_forest_pred
)

print("Random Forest Accuracy:",
      round(random_forest_accuracy, 4))

print("\nRandom Forest Report:")
print(classification_report(
    y_test,
    random_forest_pred
))


# ==================================================
# 6. COMPARE MODELS
# ==================================================

print("\n" + "=" * 50)
print("MODEL COMPARISON")
print("=" * 50)

print(
    "Logistic Regression:",
    round(logistic_accuracy, 4)
)

print(
    "Random Forest:",
    round(random_forest_accuracy, 4)
)