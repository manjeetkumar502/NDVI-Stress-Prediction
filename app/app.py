import streamlit as st
import pandas as pd
import joblib
import ee
import folium

from folium.plugins import Draw
from streamlit_folium import st_folium


# ============================================================
# VEGETATION AI
# ============================================================

st.set_page_config(
    page_title="Vegetation AI",
    page_icon="🌱",
    layout="wide"
)


# ============================================================
# CONFIGURATION
# ============================================================

MODEL_PATH = "models/random_forest.joblib"
DATA_PATH = "data/processed/real_ml_features.csv"
EE_PROJECT = "vegetation-ai"


FIELD_LOCATIONS = {
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


FEATURES = [
    "NDVI", "EVI", "TCG",
    "previous_ndvi", "previous_evi",
    "ndvi_change", "evi_change",
    "B2", "B3", "B4", "B5", "B6",
    "B7", "B8", "B8A", "B9", "B11", "B12"
]

REFLECTANCE_BANDS = [
    "B2", "B3", "B4", "B5", "B6",
    "B7", "B8", "B8A", "B9", "B11", "B12"
]


# ============================================================
# LOAD MODEL AND HISTORICAL DATA
# ============================================================

try:
    model = joblib.load(MODEL_PATH)
except Exception as e:
    st.error(f"Unable to load Random Forest model: {e}")
    st.stop()

try:
    historical_df = pd.read_csv(DATA_PATH)
    historical_df["date"] = pd.to_datetime(historical_df["date"])
except Exception as e:
    st.error(f"Unable to load historical satellite data: {e}")
    st.stop()


# ============================================================
# EARTH ENGINE
# ============================================================

def initialize_earth_engine():
    try:
        ee.Initialize(project=EE_PROJECT)
        return True, None
    except Exception as e:
        return False, str(e)


ee_ready, ee_error = initialize_earth_engine()


# ============================================================
# SENTINEL-2 CLOUD / SHADOW MASK
# ============================================================

def apply_scl_mask(image):
    scl = image.select("SCL")

    mask = (
        scl.neq(3)       # cloud shadow
        .And(scl.neq(8)) # medium probability cloud
        .And(scl.neq(9)) # high probability cloud
        .And(scl.neq(10))# cirrus
        .And(scl.neq(11))# snow/ice
    )

    return image.updateMask(mask)


# ============================================================
# SENTINEL-2 PROCESSING
# ============================================================

def process_sentinel_image(image):

    reflectance = (
        image
        .select(REFLECTANCE_BANDS)
        .multiply(0.0001)
    )

    ndvi = (
        reflectance
        .normalizedDifference(["B8", "B4"])
        .rename("NDVI")
    )

    nir = reflectance.select("B8")
    red = reflectance.select("B4")
    blue = reflectance.select("B2")

    evi = (
        nir
        .subtract(red)
        .multiply(2.5)
        .divide(
            nir
            .add(red.multiply(6))
            .subtract(blue.multiply(7.5))
            .add(1)
        )
        .rename("EVI")
    )

    tcg = (
        reflectance.select("B2").multiply(-0.2941)
        .add(reflectance.select("B3").multiply(-0.2430))
        .add(reflectance.select("B4").multiply(-0.5424))
        .add(reflectance.select("B8").multiply(0.7276))
        .add(reflectance.select("B11").multiply(0.0713))
        .add(reflectance.select("B12").multiply(-0.1608))
        .rename("TCG")
    )

    return (
        reflectance
        .addBands(ndvi)
        .addBands(evi)
        .addBands(tcg)
    )


# ============================================================
# GET LIVE FIELD DATA
# ============================================================

def get_live_field_data(field_geometry):

    collection = (
        ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
        .filterBounds(field_geometry)
        .filterDate("2024-01-01", "2027-01-01")
        .filter(
            ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", 20)
        )
        .sort("system:time_start", False)
    )

    collection_info = (
        collection
        .limit(30)
        .getInfo()
    )

    image_list = collection_info.get("features", [])

    if not image_list:
        return None, None

    selected_images = []
    selected_dates = set()

    for image_info in image_list:

        properties = image_info.get("properties", {})

        timestamp = properties.get("system:time_start")

        if timestamp is None:
            continue

        date = pd.to_datetime(
            timestamp,
            unit="ms"
        ).strftime("%Y-%m-%d")

        if date in selected_dates:
            continue

        image_id = image_info.get("id")

        if image_id is None:
            continue

        selected_dates.add(date)

        selected_images.append(
            {
                "id": image_id,
                "date": date,
                "cloud": properties.get(
                    "CLOUDY_PIXEL_PERCENTAGE"
                )
            }
        )

        if len(selected_images) >= 8:
            break

    if not selected_images:
        return None, None

    results = []

    for image_info in selected_images:

        image = ee.Image(image_info["id"])

        masked_image = apply_scl_mask(image)

        processed_image = process_sentinel_image(
            masked_image
        )

        values = (
            processed_image
            .reduceRegion(
                reducer=ee.Reducer.mean(),
                geometry=field_geometry,
                scale=20,
                bestEffort=True,
                maxPixels=10000000
            )
            .getInfo()
        )

        if not values:
            continue

        row = {
            "date": image_info["date"],
            "cloud_percentage": image_info["cloud"]
        }

        for band in REFLECTANCE_BANDS:
            row[band] = values.get(band)

        row["NDVI"] = values.get("NDVI")
        row["EVI"] = values.get("EVI")
        row["TCG"] = values.get("TCG")

        results.append(row)

    if not results:
        return None, None

    live_df = pd.DataFrame(results)

    live_df["date"] = pd.to_datetime(live_df["date"])

    live_df = (
        live_df
        .sort_values("date")
        .drop_duplicates("date")
        .reset_index(drop=True)
    )

    live_df["previous_ndvi"] = pd.NA
    live_df["previous_evi"] = pd.NA
    live_df["ndvi_change"] = pd.NA
    live_df["evi_change"] = pd.NA

    for i in range(1, len(live_df)):

        if (
            pd.notna(live_df.loc[i, "NDVI"])
            and pd.notna(live_df.loc[i - 1, "NDVI"])
        ):
            live_df.loc[i, "previous_ndvi"] = (
                live_df.loc[i - 1, "NDVI"]
            )
            live_df.loc[i, "ndvi_change"] = (
                live_df.loc[i, "NDVI"]
                - live_df.loc[i - 1, "NDVI"]
            )

        if (
            pd.notna(live_df.loc[i, "EVI"])
            and pd.notna(live_df.loc[i - 1, "EVI"])
        ):
            live_df.loc[i, "previous_evi"] = (
                live_df.loc[i - 1, "EVI"]
            )
            live_df.loc[i, "evi_change"] = (
                live_df.loc[i, "EVI"]
                - live_df.loc[i - 1, "EVI"]
            )

    if len(live_df) == 1:

        live_df.loc[0, "previous_ndvi"] = live_df.loc[0, "NDVI"]
        live_df.loc[0, "previous_evi"] = live_df.loc[0, "EVI"]
        live_df.loc[0, "ndvi_change"] = 0.0
        live_df.loc[0, "evi_change"] = 0.0

    return live_df, selected_images


# ============================================================
# FIELD AREA
# ============================================================

def get_field_area_hectares(field_geometry):

    area_m2 = field_geometry.area(maxError=1).getInfo()

    return area_m2 / 10000


# ============================================================
# NDVI TILE
# ============================================================

def get_ndvi_tile_url(image_id):

    image = ee.Image(image_id)

    masked_image = apply_scl_mask(image)

    processed_image = process_sentinel_image(
        masked_image
    )

    ndvi = processed_image.select("NDVI")

    map_info = ndvi.getMapId(
        {
            "min": -0.1,
            "max": 0.8,
            "palette": [
                "red",
                "orange",
                "yellow",
                "lightgreen",
                "green",
                "darkgreen"
            ]
        }
    )

    return (
        "https://earthengine.googleapis.com/map/"
        f"{map_info['mapid']}/{{z}}/{{x}}/{{y}}"
        f"?token={map_info['token']}"
    )


# ============================================================
# FIELD MAP
# ============================================================

def create_field_map(ndvi_tile_url=None):

    field_map = folium.Map(
        location=[28.4595, 77.0266],
        zoom_start=12,
        tiles="OpenStreetMap"
    )

    folium.TileLayer(
        tiles=(
            "https://server.arcgisonline.com/"
            "ArcGIS/rest/services/World_Imagery/"
            "MapServer/tile/{z}/{y}/{x}"
        ),
        attr="Esri",
        name="Satellite",
        overlay=False,
        control=True
    ).add_to(field_map)

    if ndvi_tile_url:

        folium.TileLayer(
            tiles=ndvi_tile_url,
            attr="Google Earth Engine",
            name="🌱 NDVI Health",
            overlay=True,
            control=True,
            opacity=0.70
        ).add_to(field_map)

    Draw(
        export=False,
        draw_options={
            "polyline": False,
            "polygon": True,
            "rectangle": True,
            "circle": False,
            "marker": False,
            "circlemarker": False
        },
        edit_options={
            "edit": True,
            "remove": True
        }
    ).add_to(field_map)

    folium.LayerControl().add_to(field_map)

    return field_map


# ============================================================
# SESSION STATE
# ============================================================

if "selected_geometry" not in st.session_state:
    st.session_state["selected_geometry"] = None

if "live_df" not in st.session_state:
    st.session_state["live_df"] = None

if "field_area" not in st.session_state:
    st.session_state["field_area"] = None

if "latest_image_id" not in st.session_state:
    st.session_state["latest_image_id"] = None


# ============================================================
# TITLE
# ============================================================

st.title("🌱 Vegetation AI")

st.subheader(
    "AI-Powered Satellite Vegetation Stress Monitoring"
)

st.write(
    "Monitor agricultural fields using real Sentinel-2 "
    "satellite imagery, Google Earth Engine and a "
    "Random Forest machine learning model."
)


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.header("🌾 Field Analysis")

analysis_mode = st.sidebar.radio(
    "Analysis Mode",
    [
        "Live Satellite Analysis",
        "Historical Analysis"
    ]
)


# ============================================================
# LIVE ANALYSIS
# ============================================================

if analysis_mode == "Live Satellite Analysis":

    st.sidebar.subheader("Field Selection")

    st.sidebar.info(
        "Draw your field boundary on the map. "
        "The complete selected area will be analyzed."
    )

    reference_field = st.sidebar.selectbox(
        "Reference location",
        list(FIELD_LOCATIONS.keys())
    )

    ref_lat, ref_lon = FIELD_LOCATIONS[
        reference_field
    ]

    st.sidebar.caption(
        f"Reference: {ref_lat:.6f}, {ref_lon:.6f}"
    )

    st.sidebar.divider()

    st.sidebar.write("Model: Random Forest")
    st.sidebar.write("Satellite: Sentinel-2")
    st.sidebar.write("Features: 18")
    st.sidebar.write("Analysis: Field-level")


    # --------------------------------------------------------
    # MAP
    # --------------------------------------------------------

    st.header("🗺️ Select Your Field")

    st.write(
        "Pan and zoom to your farm. Use the polygon or "
        "rectangle tool to draw the field boundary."
    )

    ndvi_tile_url = None

    if (
        st.session_state["latest_image_id"]
        and ee_ready
    ):
        try:
            ndvi_tile_url = get_ndvi_tile_url(
                st.session_state["latest_image_id"]
            )
        except Exception:
            ndvi_tile_url = None

    field_map = create_field_map(
        ndvi_tile_url
    )

    map_data = st_folium(
        field_map,
        width=None,
        height=550
    )

    last_drawing = None

    if map_data:
        last_drawing = map_data.get(
            "last_active_drawing"
        )

    if last_drawing:

        geometry = last_drawing.get("geometry")

        if geometry and geometry.get("type") in [
            "Polygon",
            "MultiPolygon"
        ]:

            st.session_state[
                "selected_geometry"
            ] = geometry

            st.success(
                "✅ Field boundary selected."
            )


    # --------------------------------------------------------
    # FIELD INFORMATION
    # --------------------------------------------------------

    selected_geometry = st.session_state[
        "selected_geometry"
    ]

    if selected_geometry:

        geometry_type = selected_geometry.get(
            "type"
        )

        try:

            if geometry_type == "Polygon":

                ee_geometry = ee.Geometry.Polygon(
                    selected_geometry["coordinates"]
                )

            else:

                ee_geometry = ee.Geometry.MultiPolygon(
                    selected_geometry["coordinates"]
                )

            if ee_ready:

                try:

                    area = get_field_area_hectares(
                        ee_geometry
                    )

                    st.session_state[
                        "field_area"
                    ] = area

                except Exception as e:

                    st.warning(
                        f"Could not calculate field area: {e}"
                    )

            st.divider()

            col1, col2, col3 = st.columns(3)

            with col1:
                if st.session_state["field_area"] is not None:
                    st.metric(
                        "Field Area",
                        f"{st.session_state['field_area']:.2f} ha"
                    )
                else:
                    st.metric("Field Area", "N/A")

            with col2:
                st.metric(
                    "Selection",
                    "Drawn Field"
                )

            with col3:
                st.metric(
                    "Analysis",
                    "Field-level"
                )


            # ------------------------------------------------
            # ANALYZE
            # ------------------------------------------------

            if st.button(
                "🛰️ Analyze Selected Field",
                type="primary",
                use_container_width=True
            ):

                if not ee_ready:

                    st.error(
                        "Google Earth Engine could not be initialized."
                    )

                    st.code(ee_error)

                    st.stop()

                with st.spinner(
                    "🛰️ Fetching Sentinel-2 imagery and "
                    "analyzing the complete field..."
                ):

                    try:

                        live_df, selected_images = (
                            get_live_field_data(
                                ee_geometry
                            )
                        )

                    except Exception as e:

                        st.error(
                            f"Earth Engine analysis failed: {e}"
                        )

                        st.stop()

                if live_df is None or live_df.empty:

                    st.warning(
                        "No suitable Sentinel-2 imagery "
                        "was found for this field."
                    )

                    st.stop()

                st.session_state[
                    "live_df"
                ] = live_df

                if selected_images:

                    st.session_state[
                        "latest_image_id"
                    ] = selected_images[-1]["id"]

                st.success(
                    "✅ Field analysis completed."
                )

        except Exception as e:

            st.error(
                f"Could not process selected field: {e}"
            )


    else:

        st.info(
            "👆 Draw a polygon or rectangle around your "
            "field on the map to begin."
        )


    # ========================================================
    # LIVE RESULTS
    # ========================================================

    live_df = st.session_state["live_df"]

    if live_df is not None and not live_df.empty:

        latest = live_df.iloc[-1]

        missing_features = [
            feature
            for feature in FEATURES
            if pd.isna(latest.get(feature))
        ]

        if missing_features:

            st.error(
                "Some satellite features could not be calculated."
            )

            st.write(missing_features)

            st.stop()


        X_live = pd.DataFrame(
            [
                [
                    latest[feature]
                    for feature in FEATURES
                ]
            ],
            columns=FEATURES
        )

        prediction = model.predict(X_live)[0]

        try:

            probabilities = model.predict_proba(
                X_live
            )[0]

            probability_df = pd.DataFrame(
                {
                    "Condition": model.classes_,
                    "Probability": probabilities
                }
            )

            confidence = probabilities.max() * 100

        except Exception:

            probability_df = None
            confidence = None


        # ----------------------------------------------------
        # RESULT HEADER
        # ----------------------------------------------------

        st.divider()

        st.header("🌾 Field Analysis Result")

        if st.session_state["field_area"] is not None:

            st.caption(
                f"Analyzed field area: "
                f"{st.session_state['field_area']:.2f} hectares"
            )


        # ----------------------------------------------------
        # METRICS
        # ----------------------------------------------------

        col1, col2, col3, col4, col5 = st.columns(5)

        with col1:
            st.metric(
                "Latest NDVI",
                f"{latest['NDVI']:.3f}"
            )

        with col2:
            st.metric(
                "Latest EVI",
                f"{latest['EVI']:.3f}"
            )

        with col3:
            st.metric(
                "NDVI Change",
                f"{latest['ndvi_change']:.3f}"
            )

        with col4:
            st.metric(
                "Observation Date",
                latest["date"].strftime("%d %b %Y")
            )

        with col5:

            cloud = latest["cloud_percentage"]

            if pd.notna(cloud):
                st.metric(
                    "Scene Cloud",
                    f"{cloud:.1f}%"
                )
            else:
                st.metric(
                    "Scene Cloud",
                    "N/A"
                )


        # ----------------------------------------------------
        # CONDITION
        # ----------------------------------------------------

        st.divider()

        st.header("🌿 Vegetation Condition")

        if prediction == "Stable":

            st.success(
                "🌱 Stable — vegetation condition appears stable."
            )

        elif prediction == "Moderate Stress":

            st.warning(
                "⚠️ Moderate Stress — vegetation shows "
                "signs of stress."
            )

        elif prediction == "High Stress":

            st.error(
                "🚨 High Stress — vegetation shows strong "
                "stress indicators."
            )

        else:

            st.info(
                f"Predicted condition: {prediction}"
            )


        if confidence is not None:

            st.metric(
                "Model Probability",
                f"{confidence:.1f}%"
            )

            st.caption(
                "This is the Random Forest class probability, "
                "not a guarantee of actual crop condition."
            )


        # ----------------------------------------------------
        # PROBABILITY
        # ----------------------------------------------------

        if probability_df is not None:

            st.subheader("Prediction Probability")

            probability_display = probability_df.copy()

            probability_display["Probability"] = (
                probability_display["Probability"] * 100
            ).round(2)

            probability_display = (
                probability_display
                .sort_values(
                    "Probability",
                    ascending=False
                )
                .reset_index(drop=True)
            )

            st.dataframe(
                probability_display,
                use_container_width=True,
                hide_index=True
            )


        # ----------------------------------------------------
        # NDVI LEGEND
        # ----------------------------------------------------

        st.divider()

        st.header("🗺️ NDVI Health Map")

        st.write(
            "The map above uses the latest available Sentinel-2 "
            "observation. Red/orange represents lower vegetation "
            "signal, while green represents healthier vegetation."
        )

        st.info(
            "If the NDVI layer is visible in the map, use the "
            "Layer control in the top-right corner to switch "
            "between Satellite and NDVI Health."
        )


        # ----------------------------------------------------
        # MEASUREMENTS
        # ----------------------------------------------------

        st.divider()

        st.header("🛰️ Field Satellite Measurements")

        measurement_columns = [
            "NDVI", "EVI", "TCG",
            "ndvi_change", "evi_change",
            "B2", "B3", "B4", "B5", "B6",
            "B7", "B8", "B8A", "B9", "B11", "B12"
        ]

        measurement_data = pd.DataFrame(
            {
                "Feature": measurement_columns,
                "Value": [
                    latest[column]
                    for column in measurement_columns
                ]
            }
        )

        measurement_data["Value"] = (
            measurement_data["Value"].round(5)
        )

        st.dataframe(
            measurement_data,
            use_container_width=True,
            hide_index=True
        )


        # ----------------------------------------------------
        # OBSERVATIONS
        # ----------------------------------------------------

        st.divider()

        st.header(
            "📅 Recent Sentinel-2 Field Observations"
        )

        display_live = live_df[
            [
                "date",
                "NDVI",
                "EVI",
                "TCG",
                "ndvi_change",
                "evi_change",
                "cloud_percentage"
            ]
        ].copy()

        display_live["date"] = (
            display_live["date"]
            .dt.strftime("%d %b %Y")
        )

        st.dataframe(
            display_live.sort_values(
                "date",
                ascending=False
            ),
            use_container_width=True,
            hide_index=True
        )


        # ----------------------------------------------------
        # NDVI TREND
        # ----------------------------------------------------

        st.divider()

        st.header("📈 Field NDVI Trend")

        st.line_chart(
            live_df[
                ["date", "NDVI"]
            ].set_index("date")
        )


        # ----------------------------------------------------
        # EVI TREND
        # ----------------------------------------------------

        st.header("📊 Field EVI Trend")

        st.line_chart(
            live_df[
                ["date", "EVI"]
            ].set_index("date")
        )


        # ----------------------------------------------------
        # FARMER SUMMARY
        # ----------------------------------------------------

        st.divider()

        st.header("🧑‍🌾 Field Summary")

        if prediction == "Stable":

            st.write(
                "The latest satellite observations indicate "
                "relatively stable vegetation conditions across "
                "the analyzed field."
            )

        elif prediction == "Moderate Stress":

            st.write(
                "The latest satellite observations show "
                "indicators associated with moderate vegetation "
                "stress. The field should be monitored closely."
            )

        elif prediction == "High Stress":

            st.write(
                "The latest satellite observations show strong "
                "vegetation stress indicators. Further field "
                "inspection is recommended."
            )

        st.caption(
            "Satellite analysis should support, not replace, "
            "physical field inspection."
        )


# ============================================================
# HISTORICAL ANALYSIS
# ============================================================

else:

    st.sidebar.subheader("Historical Dataset")

    fields = sorted(
        historical_df["field_id"].unique()
    )

    selected_field = st.sidebar.selectbox(
        "Select a field",
        fields
    )

    field_df = (
        historical_df[
            historical_df["field_id"] == selected_field
        ]
        .sort_values("date")
        .reset_index(drop=True)
    )

    if field_df.empty:

        st.warning(
            "No historical observations available."
        )

        st.stop()

    latest = field_df.iloc[-1]

    X_latest = pd.DataFrame(
        [
            latest[FEATURES].values
        ],
        columns=FEATURES
    )

    prediction = model.predict(X_latest)[0]

    try:

        probabilities = model.predict_proba(
            X_latest
        )[0]

        probability_df = pd.DataFrame(
            {
                "Condition": model.classes_,
                "Probability": probabilities
            }
        )

        confidence = probabilities.max() * 100

    except Exception:

        probability_df = None
        confidence = None


    st.divider()

    st.header(
        f"📍 Historical Analysis — {selected_field}"
    )

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            "Latest NDVI",
            f"{latest['NDVI']:.3f}"
        )

    with col2:
        st.metric(
            "Latest EVI",
            f"{latest['EVI']:.3f}"
        )

    with col3:
        st.metric(
            "NDVI Change",
            f"{latest['ndvi_change']:.3f}"
        )

    with col4:
        st.metric(
            "Observation Date",
            latest["date"].strftime("%d %b %Y")
        )


    st.divider()

    st.header("🌿 Vegetation Condition")

    if prediction == "Stable":

        st.success(
            "🌱 Stable — vegetation condition appears stable."
        )

    elif prediction == "Moderate Stress":

        st.warning(
            "⚠️ Moderate Stress — vegetation shows signs of stress."
        )

    elif prediction == "High Stress":

        st.error(
            "🚨 High Stress — vegetation shows strong stress indicators."
        )

    else:

        st.info(
            f"Predicted condition: {prediction}"
        )


    if confidence is not None:

        st.metric(
            "Model Probability",
            f"{confidence:.1f}%"
        )

        st.caption(
            "This represents the Random Forest class probability."
        )


    if probability_df is not None:

        st.subheader("Prediction Probability")

        probability_display = probability_df.copy()

        probability_display["Probability"] = (
            probability_display["Probability"] * 100
        ).round(2)

        probability_display = (
            probability_display
            .sort_values(
                "Probability",
                ascending=False
            )
            .reset_index(drop=True)
        )

        st.dataframe(
            probability_display,
            use_container_width=True,
            hide_index=True
        )


    st.divider()

    st.header("📈 NDVI Trend")

    st.line_chart(
        field_df[
            ["date", "NDVI"]
        ].set_index("date")
    )


    st.header("📊 EVI Trend")

    st.line_chart(
        field_df[
            ["date", "EVI"]
        ].set_index("date")
    )


    st.divider()

    st.header(
        "🛰️ Historical Satellite Observations"
    )

    recent_data = (
        field_df[
            [
                "date",
                "NDVI",
                "EVI",
                "TCG",
                "ndvi_change",
                "evi_change"
            ]
        ]
        .sort_values(
            "date",
            ascending=False
        )
        .head(10)
        .copy()
    )

    recent_data["date"] = (
        recent_data["date"]
        .dt.strftime("%d %b %Y")
    )

    st.dataframe(
        recent_data,
        use_container_width=True,
        hide_index=True
    )


# ============================================================
# SIDEBAR INFORMATION
# ============================================================

st.sidebar.divider()

st.sidebar.subheader("Dataset Information")

st.sidebar.write(
    f"Historical fields: "
    f"{historical_df['field_id'].nunique()}"
)

st.sidebar.write(
    f"Historical observations: "
    f"{len(historical_df)}"
)

st.sidebar.write(
    "ML model: Random Forest"
)

st.sidebar.write(
    "Satellite: Sentinel-2"
)

st.sidebar.write(
    "Platform: Google Earth Engine"
)


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "Vegetation AI | Sentinel-2 + "
    "Google Earth Engine + Machine Learning"
)
