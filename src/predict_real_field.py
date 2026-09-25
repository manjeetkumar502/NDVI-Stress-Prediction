import pandas as pd
import joblib


# ============================================================
# VEGETATION AI - REAL FIELD ML PREDICTION
# ============================================================

print("=" * 60)
print("VEGETATION AI - REAL FIELD ML PREDICTION")
print("=" * 60)


# ============================================================
# 1. LOAD MODEL
# ============================================================

model_path = "models/random_forest.joblib"

model = joblib.load(model_path)

print("\nRandom Forest model loaded successfully.")


# ============================================================
# 2. LOAD REAL SENTINEL-2 DATA
# ============================================================

data_path = "data/processed/final_real_dataset.csv"

df = pd.read_csv(data_path)

print("\nReal dataset shape:")
print(df.shape)


# ============================================================
# 3. FEATURES
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
# 4. CHECK FEATURES
# ============================================================

missing_features = [
    feature
    for feature in features
    if feature not in df.columns
]

if missing_features:

    print("\nMissing features:")
    print(missing_features)

    raise ValueError(
        "Required ML features are missing."
    )


# ============================================================
# 5. CREATE ML INPUT
# ============================================================

X_real = df[features]


# ============================================================
# 6. MAKE PREDICTIONS
# ============================================================

predictions = model.predict(X_real)

df["ml_prediction"] = predictions


# ============================================================
# 7. PREDICTION PROBABILITY
# ============================================================

probabilities = model.predict_proba(X_real)

df["prediction_confidence"] = (
    probabilities.max(axis=1)
)


# ============================================================
# 8. DISPLAY PREDICTION DISTRIBUTION
# ============================================================

print("\n" + "=" * 60)
print("ML PREDICTION DISTRIBUTION")
print("=" * 60)

print(
    df["ml_prediction"]
    .value_counts()
)


# ============================================================
# 9. FIELD-WISE PREDICTIONS
# ============================================================

print("\n" + "=" * 60)
print("FIELD-WISE ML PREDICTIONS")
print("=" * 60)

field_summary = pd.crosstab(
    df["field_id"],
    df["ml_prediction"]
)

print(field_summary)


# ============================================================
# 10. AVERAGE CONFIDENCE
# ============================================================

print("\nAverage prediction confidence:")

print(
    round(
        df["prediction_confidence"].mean(),
        4
    )
)


# ============================================================
# 11. SHOW SAMPLE PREDICTIONS
# ============================================================

print("\nSample predictions:")

print(
    df[
        [
            "field_id",
            "date",
            "NDVI",
            "EVI",
            "ndvi_change",
            "ml_prediction",
            "prediction_confidence"
        ]
    ].head(20).to_string(index=False)
)


# ============================================================
# 12. SAVE RESULTS
# ============================================================

output_path = (
    "data/processed/"
    "real_ml_predictions.csv"
)

df.to_csv(
    output_path,
    index=False
)


print("\nSaved predictions to:")
print(output_path)


print("\n" + "=" * 60)
print("REAL FIELD ML PREDICTION COMPLETE")
print("=" * 60)