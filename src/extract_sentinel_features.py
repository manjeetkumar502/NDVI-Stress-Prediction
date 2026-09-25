import ee

# ---------------------------------------------------------
# VEGETATION AI - REAL SENTINEL-2 FEATURE EXTRACTION
# ---------------------------------------------------------

ee.Initialize(project="vegetation-ai")

# ---------------------------------------------------------
# 1. TEST LOCATION
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
# 3. SELECT THE CLEAREST IMAGE
# ---------------------------------------------------------

image = collection.sort("CLOUDY_PIXEL_PERCENTAGE").first()


# ---------------------------------------------------------
# 4. CALCULATE NDVI
# ---------------------------------------------------------

ndvi = image.normalizedDifference(["B8", "B4"]).rename("NDVI")


# ---------------------------------------------------------
# 5. CALCULATE EVI
# ---------------------------------------------------------

evi = image.expression(
    "2.5 * ((NIR - RED) / (NIR + 6 * RED - 7.5 * BLUE + 1))",
    {
        "NIR": image.select("B8"),
        "RED": image.select("B4"),
        "BLUE": image.select("B2")
    }
).rename("EVI")


# ---------------------------------------------------------
# 6. COMBINE SATELLITE FEATURES
# ---------------------------------------------------------

features = image.select([
    "B2", "B3", "B4", "B5",
    "B6", "B7", "B8", "B8A",
    "B9", "B11", "B12"
]).addBands([ndvi, evi])


# ---------------------------------------------------------
# 7. EXTRACT VALUES AT LOCATION
# ---------------------------------------------------------

result = features.reduceRegion(
    reducer=ee.Reducer.mean(),
    geometry=point,
    scale=10,
    maxPixels=1e9
).getInfo()


# ---------------------------------------------------------
# 8. DISPLAY RESULTS
# ---------------------------------------------------------

print("=" * 60)
print("VEGETATION AI - REAL SENTINEL-2 FEATURES")
print("=" * 60)

print("\nLocation:")
print("Latitude:", latitude)
print("Longitude:", longitude)

print("\nSatellite image:")
print(image.id().getInfo())

print("\nExtracted features:")
print("-" * 60)

for feature, value in result.items():
    print(f"{feature:6} : {value}")

print("\n" + "=" * 60)
print("FEATURE EXTRACTION COMPLETE")
print("=" * 60)