import ee
import pandas as pd

# ---------------------------------------------------------
# VEGETATION AI - REAL SENTINEL-2 TIME SERIES
# ---------------------------------------------------------

# Connect to Google Earth Engine
ee.Initialize(project="vegetation-ai")


# ---------------------------------------------------------
# 1. FIELD LOCATION
# ---------------------------------------------------------

latitude = 28.4595
longitude = 77.0266

point = ee.Geometry.Point([longitude, latitude])


# ---------------------------------------------------------
# 2. GET SENTINEL-2 IMAGES
# ---------------------------------------------------------

collection = (
    ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
    .filterBounds(point)
    .filterDate("2024-01-01", "2024-12-31")
    .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", 20))
)


# ---------------------------------------------------------
# 3. CALCULATE NDVI AND EVI FOR EACH IMAGE
# ---------------------------------------------------------

def calculate_features(image):

    # Sentinel-2 reflectance scaling
    reflectance = image.select([
        "B2", "B3", "B4", "B5",
        "B6", "B7", "B8", "B8A",
        "B9", "B11", "B12"
    ]).multiply(0.0001)

    # NDVI
    ndvi = reflectance.normalizedDifference(
        ["B8", "B4"]
    ).rename("NDVI")

    # EVI
    evi = reflectance.expression(
        "2.5 * ((NIR - RED) / "
        "(NIR + 6 * RED - 7.5 * BLUE + 1))",
        {
            "NIR": reflectance.select("B8"),
            "RED": reflectance.select("B4"),
            "BLUE": reflectance.select("B2")
        }
    ).rename("EVI")

    # Combine bands + vegetation indices
    return reflectance.addBands([
        ndvi,
        evi
    ]).copyProperties(
        image,
        ["system:time_start", "CLOUDY_PIXEL_PERCENTAGE"]
    )


collection = collection.map(calculate_features)


# ---------------------------------------------------------
# 4. EXTRACT VALUES FOR OUR LOCATION
# ---------------------------------------------------------

def extract_values(image):

    values = image.reduceRegion(
        reducer=ee.Reducer.mean(),
        geometry=point,
        scale=10,
        maxPixels=1e9
    )

    return ee.Feature(
        None,
        values.set(
            "date",
            image.date().format("YYYY-MM-dd")
        ).set(
            "cloud_percentage",
            image.get("CLOUDY_PIXEL_PERCENTAGE")
        )
    )


features = collection.map(extract_values)


# ---------------------------------------------------------
# 5. SEND DATA TO PYTHON
# ---------------------------------------------------------

data = features.getInfo()["features"]


# ---------------------------------------------------------
# 6. CREATE DATAFRAME
# ---------------------------------------------------------

rows = []

for feature in data:

    properties = feature["properties"]

    rows.append(properties)


df = pd.DataFrame(rows)


# ---------------------------------------------------------
# 7. SORT BY DATE
# ---------------------------------------------------------

df["date"] = pd.to_datetime(df["date"])

df = df.sort_values("date")

df = df.reset_index(drop=True)


# ---------------------------------------------------------
# 8. CALCULATE TEMPORAL FEATURES
# ---------------------------------------------------------

df["previous_ndvi"] = df["NDVI"].shift(1)

df["previous_evi"] = df["EVI"].shift(1)

df["ndvi_change"] = (
    df["NDVI"] - df["previous_ndvi"]
)

df["evi_change"] = (
    df["EVI"] - df["previous_evi"]
)


# ---------------------------------------------------------
# 9. DISPLAY RESULTS
# ---------------------------------------------------------

print("=" * 60)
print("VEGETATION AI - REAL SENTINEL-2 TIME SERIES")
print("=" * 60)

print("\nLocation:")
print("Latitude:", latitude)
print("Longitude:", longitude)

print("\nNumber of satellite observations:")
print(len(df))

print("\nDate range:")

if len(df) > 0:
    print(df["date"].min())
    print("to")
    print(df["date"].max())

print("\nTime-series data:")
print("-" * 60)

print(df[
    [
        "date",
        "NDVI",
        "EVI",
        "previous_ndvi",
        "previous_evi",
        "ndvi_change",
        "evi_change",
        "cloud_percentage"
    ]
].to_string(index=False))


# ---------------------------------------------------------
# 10. SAVE DATA
# ---------------------------------------------------------

output_path = "data/processed/real_sentinel_timeseries.csv"

df.to_csv(output_path, index=False)

print("\n")
print("Saved to:")
print(output_path)

print("\n" + "=" * 60)
print("TIME SERIES EXTRACTION COMPLETE")
print("=" * 60)