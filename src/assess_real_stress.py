import pandas as pd

# ---------------------------------------------------------
# VEGETATION AI - REAL FIELD STRESS ASSESSMENT
# ---------------------------------------------------------

input_path = "data/processed/final_real_dataset.csv"
output_path = "data/processed/real_stress_assessment.csv"

print("=" * 60)
print("VEGETATION AI - REAL FIELD STRESS ASSESSMENT")
print("=" * 60)


# ---------------------------------------------------------
# 1. LOAD REAL DATA
# ---------------------------------------------------------

df = pd.read_csv(input_path)

df["date"] = pd.to_datetime(df["date"])

print("\nDataset shape:")
print(df.shape)

print("\nNumber of fields:")
print(df["field_id"].nunique())


# ---------------------------------------------------------
# 2. CALCULATE STRESS SCORE
# ---------------------------------------------------------

def calculate_stress(row):

    score = 0
    reasons = []

    # -----------------------------------------------------
    # NDVI CONDITION
    # -----------------------------------------------------

    if row["NDVI"] < 0.10:
        score += 2
        reasons.append("Very Low NDVI")

    elif row["NDVI"] < 0.20:
        score += 1
        reasons.append("Low NDVI")


    # -----------------------------------------------------
    # EVI CONDITION
    # -----------------------------------------------------

    if row["EVI"] < 0.05:
        score += 2
        reasons.append("Very Low EVI")

    elif row["EVI"] < 0.10:
        score += 1
        reasons.append("Low EVI")


    # -----------------------------------------------------
    # NDVI CHANGE
    # -----------------------------------------------------

    if row["ndvi_change"] < -0.05:
        score += 2
        reasons.append("Sharp NDVI Decline")

    elif row["ndvi_change"] < -0.02:
        score += 1
        reasons.append("NDVI Decline")


    # -----------------------------------------------------
    # EVI CHANGE
    # -----------------------------------------------------

    if row["evi_change"] < -0.03:
        score += 2
        reasons.append("Sharp EVI Decline")

    elif row["evi_change"] < -0.01:
        score += 1
        reasons.append("EVI Decline")


    # -----------------------------------------------------
    # RETURN SCORE + REASONS
    # -----------------------------------------------------

    return pd.Series(
        [
            score,
            ", ".join(reasons) if reasons else "Normal vegetation"
        ]
    )


stress_results = df.apply(
    calculate_stress,
    axis=1
)

df["stress_score"] = stress_results[0]
df["stress_reasons"] = stress_results[1]


# ---------------------------------------------------------
# 3. CLASSIFY STRESS
# ---------------------------------------------------------

def classify_stress(score):

    if score >= 5:
        return "High Stress"

    elif score >= 2:
        return "Moderate Stress"

    else:
        return "Stable"


df["stress_condition"] = (
    df["stress_score"]
    .apply(classify_stress)
)


# ---------------------------------------------------------
# 4. STRESS DISTRIBUTION
# ---------------------------------------------------------

print("\nStress distribution:")

stress_distribution = (
    df["stress_condition"]
    .value_counts()
)

print(stress_distribution)


# ---------------------------------------------------------
# 5. STRESS PERCENTAGE
# ---------------------------------------------------------

print("\nStress percentage:")

stress_percentage = (
    df["stress_condition"]
    .value_counts(normalize=True)
    * 100
)

print(
    stress_percentage.round(2)
)


# ---------------------------------------------------------
# 6. FIELD-WISE STRESS DISTRIBUTION
# ---------------------------------------------------------

print("\nStress distribution by field:")

field_summary = pd.crosstab(
    df["field_id"],
    df["stress_condition"]
)

print(field_summary)


# ---------------------------------------------------------
# 7. AVERAGE STRESS SCORE BY FIELD
# ---------------------------------------------------------

print("\nAverage stress score by field:")

field_score = (
    df.groupby("field_id")["stress_score"]
    .mean()
    .sort_values(ascending=False)
)

print(field_score.round(2))


# ---------------------------------------------------------
# 8. HIGHEST STRESS OBSERVATIONS
# ---------------------------------------------------------

print("\nTop 10 highest stress observations:")

top_stress = df.sort_values(
    "stress_score",
    ascending=False
)[
    [
        "field_id",
        "date",
        "NDVI",
        "EVI",
        "ndvi_change",
        "evi_change",
        "stress_score",
        "stress_condition",
        "stress_reasons"
    ]
].head(10)

print(top_stress.to_string(index=False))


# ---------------------------------------------------------
# 9. CHECK LABEL BALANCE
# ---------------------------------------------------------

print("\nClass balance check:")

for condition in [
    "Stable",
    "Moderate Stress",
    "High Stress"
]:

    count = (
        df["stress_condition"] == condition
    ).sum()

    percentage = (
        count / len(df) * 100
    )

    print(
        f"{condition}: {count} observations "
        f"({percentage:.2f}%)"
    )


# ---------------------------------------------------------
# 10. SAVE
# ---------------------------------------------------------

df.to_csv(
    output_path,
    index=False
)

print("\nSaved to:")
print(output_path)

print("\nFinal dataset shape:")
print(df.shape)

print("\n" + "=" * 60)
print("REAL FIELD STRESS ASSESSMENT COMPLETE")
print("=" * 60)