import ee
import pandas as pd


# ============================================================
# VEGETATION AI - MULTIPLE REAL FIELD COLLECTION
# ============================================================

print("=" * 60)
print("VEGETATION AI - MULTIPLE REAL FIELD COLLECTION")
print("=" * 60)


# ============================================================
# 1. INITIALIZE EARTH ENGINE
# ============================================================

ee.Initialize(project="vegetation-ai")


# ============================================================
# 2. FIELD LOCATIONS
# ============================================================

fields = {
    "field_1": (28.4595, 77.0266),
    "field_2": (28.4700, 77.0400),
    "field_3": (28.4500, 77.0150),
    "field_4": (28.4800, 77.0550),
    "field_5": (28.4400, 77.0000),

    "field_6": (28.4650, 77.0300),
    "field_7": (28.4550, 77.0350),
    "field_8": (28.4750, 77.0250),
    "field_9": (28.4450, 77.0200),
    "field_10": (28.4350, 77.0350),
}


# ============================================================
# 3. SENTINEL-2 COLLECTION
# ============================================================

all_data = []


for field_id, (latitude, longitude) in fields.items():

    print("\nProcessing:", field_id)
    print("Latitude:", latitude)
    print("Longitude:", longitude)

    point = ee.Geometry.Point([
        longitude,
        latitude
    ])

    collection = (
        ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
        .filterBounds(point)
        .filterDate(
            "2024-01-01",
            "2025-01-01"
        )
        .filter(
            ee.Filter.lt(
                "CLOUDY_PIXEL_PERCENTAGE",
                20
            )
        )
        .sort("system:time_start")
    )

    count = collection.size().getInfo()

    print("Images found:", count)

    if count == 0:
        print("No suitable images found.")
        continue

    images = collection.toList(count)

    previous_ndvi = None
    previous_evi = None


    # ========================================================
    # 4. PROCESS EACH SATELLITE IMAGE
    # ========================================================

    for i in range(count):

        image = ee.Image(images.get(i))

        date = ee.Date(
            image.get("system:time_start")
        ).format("YYYY-MM-dd").getInfo()


        # ----------------------------------------------------
        # Spectral bands
        # ----------------------------------------------------

        bands = image.select([
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
        ]).reduceRegion(
            reducer=ee.Reducer.mean(),
            geometry=point,
            scale=20,
            bestEffort=True
        ).getInfo()


        # ----------------------------------------------------
        # NDVI
        # ----------------------------------------------------

        ndvi = image.normalizedDifference(
            ["B8", "B4"]
        ).reduceRegion(
            reducer=ee.Reducer.mean(),
            geometry=point,
            scale=20,
            bestEffort=True
        ).getInfo().get("nd")


        # ----------------------------------------------------
        # Convert Sentinel-2 bands to reflectance
        # ----------------------------------------------------
        # Sentinel-2 Surface Reflectance values are scaled
        # by 10000. EVI must use reflectance values.

        reflectance = image.select([
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
        ]).multiply(0.0001)


        # ----------------------------------------------------
        # EVI
        # ----------------------------------------------------

        evi = reflectance.expression(
            "2.5 * ((NIR - RED) / "
            "(NIR + 6 * RED - 7.5 * BLUE + 1))",
            {
                "NIR": reflectance.select("B8"),
                "RED": reflectance.select("B4"),
                "BLUE": reflectance.select("B2")
            }
        ).reduceRegion(
            reducer=ee.Reducer.mean(),
            geometry=point,
            scale=20,
            bestEffort=True
        ).getInfo().get("constant")


        # ----------------------------------------------------
        # Skip invalid observations
        # ----------------------------------------------------

        if ndvi is None or evi is None:
            continue


        # ====================================================
        # 5. TEMPORAL FEATURES
        # ====================================================

        if previous_ndvi is None:

            previous_ndvi_value = None
            previous_evi_value = None

            ndvi_change = None
            evi_change = None

        else:

            previous_ndvi_value = previous_ndvi
            previous_evi_value = previous_evi

            ndvi_change = (
                ndvi - previous_ndvi
            )

            evi_change = (
                evi - previous_evi
            )


        # ====================================================
        # 6. TCG
        # ====================================================
        # TCG calculation is performed using reflectance
        # instead of raw scaled Sentinel-2 values.

        tcg = (
            -0.2848 * bands.get("B2", 0) * 0.0001
            -0.2435 * bands.get("B3", 0) * 0.0001
            -0.5436 * bands.get("B4", 0) * 0.0001
            +0.7243 * bands.get("B8", 0) * 0.0001
            +0.0840 * bands.get("B11", 0) * 0.0001
            -0.1800 * bands.get("B12", 0) * 0.0001
        )


        # ====================================================
        # 7. CLOUD PERCENTAGE
        # ====================================================

        cloud_percentage = image.get(
            "CLOUDY_PIXEL_PERCENTAGE"
        ).getInfo()


        # ====================================================
        # 8. CREATE ROW
        # ====================================================

        row = {
            "field_id": field_id,
            "date": date,

            "NDVI": ndvi,
            "EVI": evi,
            "TCG": tcg,

            "previous_ndvi": previous_ndvi_value,
            "previous_evi": previous_evi_value,

            "ndvi_change": ndvi_change,
            "evi_change": evi_change,

            "B2": bands.get("B2"),
            "B3": bands.get("B3"),
            "B4": bands.get("B4"),
            "B5": bands.get("B5"),
            "B6": bands.get("B6"),
            "B7": bands.get("B7"),
            "B8": bands.get("B8"),
            "B8A": bands.get("B8A"),
            "B9": bands.get("B9"),
            "B11": bands.get("B11"),
            "B12": bands.get("B12"),

            "cloud_percentage": cloud_percentage
        }

        all_data.append(row)


        # ====================================================
        # 9. UPDATE PREVIOUS VALUES
        # ====================================================

        previous_ndvi = ndvi
        previous_evi = evi


# ============================================================
# 10. CREATE DATAFRAME
# ============================================================

df = pd.DataFrame(all_data)


# ============================================================
# 11. SAVE DATASET
# ============================================================

output_path = (
    "data/processed/"
    "multiple_fields_timeseries.csv"
)

df.to_csv(
    output_path,
    index=False
)


# ============================================================
# 12. SUMMARY
# ============================================================

print("\n" + "=" * 60)
print("MULTIPLE FIELD COLLECTION COMPLETE")
print("=" * 60)

print("\nTotal observations:")
print(len(df))

print("\nFields collected:")

if not df.empty:

    print(
        df["field_id"]
        .value_counts()
        .sort_index()
    )

print("\nFinal dataset shape:")
print(df.shape)

print("\nSaved to:")
print(output_path)

print("\n" + "=" * 60)