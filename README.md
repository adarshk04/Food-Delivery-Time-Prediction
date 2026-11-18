# 🍔 Food Delivery Time Prediction & Multi-Model Comparison App

An end-to-end Machine Learning regression project that predicts food delivery duration (`delivery_minutes`) using only pre-checkout order attributes, benchmarks **5 tuned regression algorithms**, and deploys an interactive **Streamlit** web application for real-time ETA comparison.

---

## 📌 Problem Statement
Online food delivery platforms often rely on flat average delivery time estimates at checkout. Because real-world delivery duration varies significantly based on kitchen preparation speed, travel distance, traffic rush hours, courier vehicle type, and weather, flat averages frequently miss the mark—leading to late deliveries, customer dissatisfaction, and refund costs.

**Goal:** Build a regression pipeline that accurately estimates delivery time in minutes at order placement, identifies the primary factors causing delays, and provides an interactive decision dashboard.

---

## 📂 Repository Structure
```text
├── Delivery_Time_Prediction.ipynb   # Complete EDA, feature engineering, GridSearchCV tuning & charts
├── app.py                           # Interactive Streamlit multi-model comparison app
├── models_dict.pkl                  # Serialized bundle of all 5 tuned models & StandardScaler
├── deliveries_train.csv             # Training dataset (8,000 orders)
├── deliveries_test.csv              # Test dataset (2,000 orders)
├── predictions.csv                  # Final test set predictions (order_id, predicted_minutes)
├── requirements.txt                 # Python dependencies
└── images/                          # Evaluation plots & Streamlit UI screenshot

## 🛠️ Methodology & Workflow
Exploratory Data Analysis (EDA): Analyzed target distribution (8 to 139 mins, mean 32.05 mins), skewed driver experience, and rare long-tail delays (>70 mins).

Leak-Free Data Cleaning: Imputed missing courier_trips_completed using the training median (147 trips) and missing weather using the training mode (clear) strictly from the training split.

Feature Engineering: Created domain-specific pre-checkout features:

distance_prep_interaction (distance_km × restaurant_avg_prep_minutes)

subtotal_per_item (order_subtotal / items_count)

is_peak_hour (Lunch rush 11:00–13:59 & Dinner rush 18:00–20:59)

order_hour, day_of_week, and is_weekend

Hyperparameter Tuning (GridSearchCV): Evaluated 5 regression models using 3-Fold Cross-Validation optimizing directly for Mean Absolute Error (MAE) with outlier-resistant objectives (Huber and L1 loss).

🖥️ Streamlit Web Application
The Streamlit app allows users to enter order details at checkout, view live delivery time predictions across all 5 models side-by-side, and see the recommended ETA from the top-performing model.