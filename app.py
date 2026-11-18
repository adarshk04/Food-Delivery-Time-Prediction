import os
import pickle
import numpy as np
import pandas as pd
import streamlit as st

# --- Page Config ---
st.set_page_config(
    page_title=" Delivery Time Predictor",
    page_icon="🍔",
    layout="wide"
)

st.title("🍔Food Delivery Time Predictor")
st.markdown(
    "Estimate delivery times from order attributes known at checkout. "
    "Compares **4 Machine Learning models** and recommends the top performer."
)


# --- Load Model Bundle ---
@st.cache_resource
def load_bundle():
    if not os.path.exists("models_dict.pkl"):
        return None
    with open("models_dict.pkl", "rb") as f:
        return pickle.load(f)


bundle = load_bundle()

if bundle is None:
    st.error("⚠️ `models_dict.pkl` not found! Please run `python train_models.py` first.")
    st.stop()

models = bundle["models"]
metrics = bundle["metrics"]
best_model_name = bundle["best_model_name"]
feature_columns = bundle["feature_columns"]
numeric_cols = bundle["numeric_cols"]
scaler = bundle["scaler"]

# --- Input Form ---
with st.form("delivery_form"):
    st.subheader("1. Order & Restaurant Information")
    c1, c2, c3 = st.columns(3)
    with c1:
        cuisine = st.selectbox("Cuisine Type", [
            "burgers", "chinese", "indian", "mexican", "pizza", "salads", "sushi", "thai"
        ])
        city_zone = st.selectbox("Delivery Zone", [
            "downtown", "midtown", "uptown", "suburbs_north", "suburbs_south"
        ])
    with c2:
        restaurant_prep = st.number_input(
            "Restaurant Avg Prep Time (min)", min_value=5.0, max_value=30.0, value=14.0, step=0.5
        )
        distance_km = st.number_input(
            "Delivery Distance (km)", min_value=0.5, max_value=25.0, value=3.5, step=0.1
        )
    with c3:
        items_count = st.number_input("Number of Items", min_value=1, max_value=20, value=2, step=1)
        order_subtotal = st.number_input("Order Subtotal ($)", min_value=5.0, max_value=300.0, value=28.50, step=1.0)

    st.subheader("2. Logistics, Courier & Timing")
    c4, c5, c6 = st.columns(3)
    with c4:
        courier_vehicle = st.selectbox("Courier Vehicle", ["scooter", "bike", "car"])
        weather = st.selectbox("Weather Condition", ["clear", "cloudy", "rain", "heavy_rain"])
    with c5:
        courier_trips = st.number_input("Courier Trips Completed", min_value=1, max_value=5000, value=150, step=10)
        order_hour = st.slider("Order Hour (24h format)", min_value=10, max_value=23, value=19)
    with c6:
        day_of_week = st.selectbox(
            "Day of Week",
            [0, 1, 2, 3, 4, 5, 6],
            format_func=lambda x: ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"][x],
            index=4
        )

    submitted = st.form_submit_button("⚡ Predict Delivery Times", use_container_width=True)

# --- Feature Transformation & Prediction ---
if submitted:
    # 1. Feature Engineering
    is_weekend = 1 if day_of_week in [5, 6] else 0
    is_peak_hour = 1 if order_hour in [11, 12, 13, 18, 19, 20] else 0
    subtotal_per_item = order_subtotal / max(items_count, 1)
    distance_prep_interaction = distance_km * restaurant_prep

    # 2. Build row matching exact training feature columns
    row = {col: 0.0 for col in feature_columns}

    # Populate raw numerical features
    raw_numeric_vals = {
        'restaurant_avg_prep_minutes': restaurant_prep,
        'distance_km': distance_km,
        'items_count': items_count,
        'order_subtotal': order_subtotal,
        'courier_trips_completed': courier_trips,
        'order_hour': order_hour,
        'day_of_week': day_of_week,
        'subtotal_per_item': subtotal_per_item,
        'distance_prep_interaction': distance_prep_interaction
    }

    # Scale numerical values
    temp_num_df = pd.DataFrame([raw_numeric_vals])[numeric_cols]
    scaled_num = scaler.transform(temp_num_df)[0]
    for idx, col in enumerate(numeric_cols):
        row[col] = scaled_num[idx]

    # Populate categorical dummy variables
    if f"cuisine_{cuisine}" in row:
        row[f"cuisine_{cuisine}"] = 1.0
    if f"city_zone_{city_zone}" in row:
        row[f"city_zone_{city_zone}"] = 1.0
    if f"courier_vehicle_{courier_vehicle}" in row:
        row[f"courier_vehicle_{courier_vehicle}"] = 1.0
    if f"weather_{weather}" in row:
        row[f"weather_{weather}"] = 1.0

    X_input = pd.DataFrame([row], columns=feature_columns)

    # 3. Predict across all 4 models
    predictions = {}
    for name, mdl in models.items():
        predictions[name] = max(round(float(mdl.predict(X_input)[0]), 1), 5.0)

    # --- UI Display ---
    st.divider()
    st.subheader("Model Predictions Comparison")

    # col_m1, col_m2, col_m3, col_m4 = st.columns(4)
    # model_cols = [col_m1, col_m2, col_m3, col_m4]
    model_cols = st.columns(len(models))

    for (name, pred_val), col in zip(predictions.items(), model_cols):
        with col:
            is_best = (name == best_model_name)
            badge = " ⭐ (Best)" if is_best else ""
            st.metric(
                label=f"{name}{badge}",
                value=f"{pred_val} mins",
                delta=f"Val MAE: {metrics[name]['MAE']:.2f}m"
            )

    # --- Best Model Recommendation Box ---
    st.success(
        f"**Recommended Delivery Time: {predictions[best_model_name]} minutes** (via **{best_model_name}**)"
    )

    with st.expander("Why was this model chosen? (Model Diagnostics)"):

      st.write(f"""
        - **Lowest Validation Error:** **{best_model_name}** achieved the lowest Mean Absolute Error (**{metrics[best_model_name]['MAE']:.4f} minutes**) and highest $R^2$ (**{metrics[best_model_name]['R2']:.4f}**) across the 1,600 validation orders, beating the flat average baseline (~7.63 minutes).
        - **Accuracy Over Speed Bias:** Although Linear Regression predicted a lower time for this single order ({predictions['Linear Regression']} mins), its overall validation error is higher ({metrics['Linear Regression']['MAE']:.2f} mins MAE). **{best_model_name}** is selected because it is the most reliable across all orders.
        - **Handles Non-Linear Interactions:** As an advanced tree-ensemble / gradient boosting method, **{best_model_name}** captures complex interactions between distance, kitchen prep time, peak hours, and severe weather (`heavy_rain`) without over-reacting to extreme outlier delays.
        """)