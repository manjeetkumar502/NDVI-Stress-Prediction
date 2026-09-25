import ee

# Connect to Earth Engine
ee.Initialize(project="vegetation-ai")

# Test location
# Example location only — we will later replace this
# with the farmer's selected location.
latitude = 28.4595
longitude = 77.0266

point = ee.Geometry.Point([longitude, latitude])

# Sentinel-2 Surface Reflectance dataset
collection = (
    ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
    .filterBounds(point)
    .filterDate("2024-01-01", "2024-12-31")
    .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", 20))
)

# Number of available images
image_count = collection.size().getInfo()

print("=" * 60)
print("VEGETATION AI - SENTINEL-2 TEST")
print("=" * 60)

print("Location:")
print("Latitude:", latitude)
print("Longitude:", longitude)

print("\nSentinel-2 images found:", image_count)

if image_count > 0:

    image = collection.sort("CLOUDY_PIXEL_PERCENTAGE").first()

    print("\nBest image selected successfully!")

    print("Image ID:")
    print(image.id().getInfo())

    print("\nAvailable bands:")
    print(image.bandNames().getInfo())

else:
    print("\nNo suitable Sentinel-2 images found.")