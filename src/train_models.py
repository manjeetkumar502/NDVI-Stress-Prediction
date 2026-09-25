import pandas as pd
import joblib
import os

from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    classification_report
)


# ============================================================
# VEGETATION AI - REAL SATELLITE ML TRAINING
# ============================================================

print("=" * 70)
print("VEGETATION AI - REAL SATELLITE ML TRAINING")
print("=" * 70)


# ============================================================
# 1. LOAD REAL ML FEATURES
# ============================================================

features_path = "data/processed/real_ml_features.csv"
stress_path = "data/processed/real_stress_assessment.csv"

features_df = pd.read_csv(features_path)
stress_df = pd.read_csv(stress_path)

print("\nReal ML feature dataset:")
print(features_df.shape)

print("\nReal stress assessment dataset:")
print(stress_df.shape)


# ============================================================
# 2. CONVERT DATE
# ============================================================

features_df["date"] = pd.to_datetime(
    features_df["date"]
)

stress_df["date"] = pd.to_datetime(
    stress_df["date"]
)


# ============================================================
# 3. MERGE FEATURES + TARGET
# ============================================================

df = pd.merge(
    features_df,
    stress_df[
        [
            "field_id",
            "date",
            "stress_score",
            "stress_condition"
        ]
    ],
    on=[
        "field_id",
        "date"
    ],
    how="inner"
)


# ============================================================
# 4. CHECK MERGED DATA
# ============================================================

print("\nMerged dataset shape:")
print(df.shape)

print("\nNumber of fields:")
print(df["field_id"].nunique())

print("\nStress distribution:")
print(
    df["stress_condition"].value_counts()
)


# ============================================================
# 5. FEATURES
# ============================================================

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


# ============================================================
# 6. REMOVE MISSING VALUES
# ============================================================

before_rows = len(df)

df = df.dropna(
    subset=features + ["stress_condition"]
).reset_index(drop=True)

after_rows = len(df)

print("\nRows removed because of missing values:")
print(before_rows - after_rows)


# ============================================================
# 7. FIELD-WISE TRAIN / TEST SPLIT
# ============================================================

fields = sorted(
    df["field_id"].unique()
)

print("\nFields available:")
print(fields)


# ------------------------------------------------------------
# IMPORTANT:
# We keep complete fields separate between training and testing.
# ------------------------------------------------------------

test_fields = [
    "field_2",
    "field_5",
    "field_8"
]

train_fields = [
    field
    for field in fields
    if field not in test_fields
]


print("\nTraining fields:")
print(train_fields)

print("\nTesting fields:")
print(test_fields)


train_df = df[
    df["field_id"].isin(train_fields)
].copy()

test_df = df[
    df["field_id"].isin(test_fields)
].copy()


# ============================================================
# 8. CREATE X AND y
# ============================================================

X_train = train_df[features]
y_train = train_df["stress_condition"]

X_test = test_df[features]
y_test = test_df["stress_condition"]


print("\nTraining data:")
print(X_train.shape)

print("\nTesting data:")
print(X_test.shape)


print("\nTraining class distribution:")
print(
    y_train.value_counts()
)


print("\nTesting class distribution:")
print(
    y_test.value_counts()
)


# ============================================================
# 9. CREATE MODELS
# ============================================================

models = {

    "Logistic Regression": Pipeline([
        (
            "scaler",
            StandardScaler()
        ),

        (
            "model",
            LogisticRegression(
                max_iter=2000,
                class_weight="balanced",
                random_state=42
            )
        )
    ]),

    "Random Forest": RandomForestClassifier(
        n_estimators=300,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1
    ),

    "Gradient Boosting": GradientBoostingClassifier(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=3,
        random_state=42
    )
}


# ============================================================
# 10. TRAIN AND EVALUATE
# ============================================================

results = []

trained_models = {}


for name, model in models.items():

    print("\n" + "=" * 70)
    print("TRAINING:", name)
    print("=" * 70)

    model.fit(
        X_train,
        y_train
    )

    predictions = model.predict(
        X_test
    )

    accuracy = accuracy_score(
        y_test,
        predictions
    )

    balanced_accuracy = balanced_accuracy_score(
        y_test,
        predictions
    )

    macro_f1 = f1_score(
        y_test,
        predictions,
        average="macro"
    )

    print("\nAccuracy:")
    print(round(accuracy, 4))

    print("\nBalanced Accuracy:")
    print(round(balanced_accuracy, 4))

    print("\nMacro F1:")
    print(round(macro_f1, 4))

    print("\nClassification Report:")

    print(
        classification_report(
            y_test,
            predictions,
            zero_division=0
        )
    )

    results.append({
        "Model": name,
        "Accuracy": accuracy,
        "Balanced Accuracy": balanced_accuracy,
        "Macro F1": macro_f1
    })

    trained_models[name] = model


# ============================================================
# 11. MODEL COMPARISON
# ============================================================

results_df = pd.DataFrame(
    results
)

results_df = results_df.sort_values(
    "Macro F1",
    ascending=False
)


print("\n" + "=" * 70)
print("MODEL COMPARISON")
print("=" * 70)

print(
    results_df.to_string(
        index=False
    )
)


# ============================================================
# 12. SELECT BEST MODEL
# ============================================================

best_model_name = (
    results_df.iloc[0]["Model"]
)

best_model = trained_models[
    best_model_name
]


print("\n" + "=" * 70)
print("BEST MODEL")
print("=" * 70)

print(
    "\nBest model:",
    best_model_name
)

print(
    "Macro F1:",
    round(
        results_df.iloc[0]["Macro F1"],
        4
    )
)

print(
    "Balanced Accuracy:",
    round(
        results_df.iloc[0]["Balanced Accuracy"],
        4
    )
)


# ============================================================
# 13. RANDOM FOREST FEATURE IMPORTANCE
# ============================================================

if "Random Forest" in trained_models:

    random_forest = trained_models[
        "Random Forest"
    ]

    importance = pd.DataFrame({
        "feature": features,
        "importance": random_forest.feature_importances_
    })

    importance = importance.sort_values(
        "importance",
        ascending=False
    )

    print("\n" + "=" * 70)
    print("RANDOM FOREST FEATURE IMPORTANCE")
    print("=" * 70)

    print(
        importance.to_string(
            index=False
        )
    )


# ============================================================
# 14. SAVE BEST MODEL
# ============================================================

os.makedirs(
    "models",
    exist_ok=True
)

model_path = "models/random_forest.joblib"

joblib.dump(
    best_model,
    model_path
)


print("\n" + "=" * 70)
print("BEST MODEL SAVED")
print("=" * 70)

print("\nModel:")
print(best_model_name)

print("\nModel path:")
print(model_path)


# ============================================================
# 15. SAVE MODEL RESULTS
# ============================================================

results_path = (
    "data/processed/real_model_results.csv"
)

results_df.to_csv(
    results_path,
    index=False
)

print("\nModel results saved to:")
print(results_path)


# ============================================================
# COMPLETE
# ============================================================

print("\n" + "=" * 70)
print("REAL SATELLITE ML TRAINING COMPLETE")
print("=" * 70)