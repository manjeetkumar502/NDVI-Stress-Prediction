import pandas as pd


print("=" * 60)
print("VEGETATION AI - REAL MODEL RELIABILITY ANALYSIS")
print("=" * 60)


# ============================================================
# 1. LOAD PREDICTIONS
# ============================================================

path = "data/processed/real_ml_predictions.csv"

df = pd.read_csv(path)

print("\nDataset shape:")
print(df.shape)


# ============================================================
# 2. COMPARE ML VS RULE-BASED ASSESSMENT
# ============================================================

# Load rule-based assessment
stress_path = "data/processed/real_stress_assessment.csv"

stress_df = pd.read_csv(stress_path)

# Keep only required columns
stress_df = stress_df[
    [
        "field_id",
        "date",
        "stress_condition"
    ]
]

# Convert dates
df["date"] = pd.to_datetime(df["date"])
stress_df["date"] = pd.to_datetime(stress_df["date"])


# Merge
comparison = df.merge(
    stress_df,
    on=["field_id", "date"],
    how="left"
)


# ============================================================
# 3. AGREEMENT
# ============================================================

comparison["agreement"] = (
    comparison["ml_prediction"]
    ==
    comparison["stress_condition"]
)

agreement = comparison["agreement"].mean()

print("\nML vs Rule-Based Agreement:")

print(
    round(agreement * 100, 2),
    "%"
)


# ============================================================
# 4. AVERAGE CONFIDENCE
# ============================================================

avg_confidence = (
    comparison["prediction_confidence"]
    .mean()
)

print("\nAverage ML confidence:")

print(
    round(avg_confidence * 100, 2),
    "%"
)


# ============================================================
# 5. LOW CONFIDENCE PREDICTIONS
# ============================================================

low_confidence = comparison[
    comparison["prediction_confidence"] < 0.60
]

print("\nLow-confidence predictions (<60%):")

print(
    len(low_confidence)
)


print("\nPercentage of low-confidence predictions:")

print(
    round(
        len(low_confidence)
        /
        len(comparison)
        *
        100,
        2
    ),
    "%"
)


# ============================================================
# 6. FIELD-WISE AGREEMENT
# ============================================================

print("\n" + "=" * 60)
print("FIELD-WISE AGREEMENT")
print("=" * 60)

field_agreement = (
    comparison
    .groupby("field_id")["agreement"]
    .mean()
    * 100
)

print(
    field_agreement.round(2)
)


# ============================================================
# 7. FIELD-WISE CONFIDENCE
# ============================================================

print("\n" + "=" * 60)
print("FIELD-WISE CONFIDENCE")
print("=" * 60)

field_confidence = (
    comparison
    .groupby("field_id")["prediction_confidence"]
    .mean()
    * 100
)

print(
    field_confidence.round(2)
)


# ============================================================
# 8. CONFUSION TABLE
# ============================================================

print("\n" + "=" * 60)
print("ML vs RULE-BASED TABLE")
print("=" * 60)

confusion = pd.crosstab(
    comparison["stress_condition"],
    comparison["ml_prediction"]
)

print(confusion)


# ============================================================
# 9. PREDICTION DISTRIBUTION
# ============================================================

print("\n" + "=" * 60)
print("ML PREDICTION DISTRIBUTION")
print("=" * 60)

print(
    comparison["ml_prediction"]
    .value_counts()
)


# ============================================================
# 10. SAVE ANALYSIS
# ============================================================

output_path = (
    "data/processed/"
    "real_prediction_analysis.csv"
)

comparison.to_csv(
    output_path,
    index=False
)

print("\nSaved detailed analysis to:")

print(output_path)


print("\n" + "=" * 60)
print("REAL MODEL RELIABILITY ANALYSIS COMPLETE")
print("=" * 60)