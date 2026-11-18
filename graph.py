import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

# 1. Load & Prepare Data
df_train = pd.read_csv('deliveries_train.csv')
median_trips = df_train['courier_trips_completed'].median()
mode_weather = df_train['weather'].mode()[0]

data = df_train.copy()
data['courier_trips_completed'] = data['courier_trips_completed'].fillna(
    median_trips
)
data['weather'] = data['weather'].fillna(mode_weather)

order_dt = pd.to_datetime(data['order_placed_at'])
data['order_hour'] = order_dt.dt.hour
data['day_of_week'] = order_dt.dt.dayofweek
data['is_weekend'] = data['day_of_week'].isin([5, 6]).astype(int)
data['is_peak_hour'] = data['order_hour'].isin([11, 12, 13, 18, 19, 20]).astype(
    int
)
data['subtotal_per_item'] = data['order_subtotal'] / data['items_count'].clip(
    lower=1
)
data['distance_prep_interaction'] = (
    data['distance_km'] * data['restaurant_avg_prep_minutes']
)

y = data['delivery_minutes']
X_raw = data.drop(columns=['order_id', 'order_placed_at', 'delivery_minutes'])
numeric_cols = [
    'restaurant_avg_prep_minutes',
    'distance_km',
    'items_count',
    'order_subtotal',
    'courier_trips_completed',
    'order_hour',
    'day_of_week',
    'subtotal_per_item',
    'distance_prep_interaction',
]
categorical_cols = ['cuisine', 'city_zone', 'courier_vehicle', 'weather']
X_encoded = pd.get_dummies(X_raw, columns=categorical_cols, drop_first=True)

X_train, X_val, y_train, y_val = train_test_split(
    X_encoded, y, test_size=0.20, random_state=42
)
scaler = StandardScaler()
X_train[numeric_cols] = scaler.fit_transform(X_train[numeric_cols])
X_val[numeric_cols] = scaler.transform(X_val[numeric_cols])

# 2. Train Models & Calculate Exact Validation MAEs
baseline_mae = mean_absolute_error(y_val, np.full(len(y_val), y_train.mean()))

models = {
    'Random Forest': RandomForestRegressor(
        n_estimators=100,
        max_depth=10,
        criterion='absolute_error',
        random_state=42,
        n_jobs=-1,
    ),
    'Linear Regression': LinearRegression(),
    'Ridge Regression': Ridge(alpha=10.0, random_state=42),
    'Gradient Boosting': GradientBoostingRegressor(
        loss='huber',
        n_estimators=100,
        learning_rate=0.1,
        max_depth=4,
        random_state=42,
    ),
    'XGBoost Regressor': XGBRegressor(
        objective='reg:absoluteerror',
        n_estimators=100,
        learning_rate=0.05,
        max_depth=3,
        random_state=42,
        n_jobs=-1,
    ),
}

maes = {'Baseline (Mean)': baseline_mae}
for name, model in models.items():
  model.fit(X_train, y_train)
  preds = model.predict(X_val)
  maes[name] = mean_absolute_error(y_val, preds)

# Sort from highest error to lowest error
sorted_models = sorted(maes.items(), key=lambda x: x[1], reverse=True)
names = [x[0] for x in sorted_models]
scores = [x[1] for x in sorted_models]
colors = [
    '#888888'
    if n == 'Baseline (Mean)'
    else ('#FF3008' if n == 'XGBoost Regressor' else '#4A90E2')
    for n in names
]

# 3. Generate & Save '2_model_comparison.png'
plt.style.use('seaborn-v0_8-whitegrid')
fig, ax = plt.subplots(figsize=(8.5, 4.8))
bars = ax.barh(names, scores, color=colors)

ax.set_title(
    'Validation Mean Absolute Error (MAE) by Model (Lower is Better)',
    fontsize=13,
    fontweight='bold',
)
ax.set_xlabel('Mean Absolute Error (Minutes)')

for bar in bars:
  width = bar.get_width()
  ax.text(
      width + 0.15,
      bar.get_y() + bar.get_height() / 2,
      f'{width:.2f} min',
      va='center',
      fontweight='bold',
  )

ax.set_xlim(0, 9.2)
plt.tight_layout()
plt.savefig('2_model_comparison.png', dpi=200)
plt.close()

# 4. Also Generate '3_feature_importance.png' & '4_actual_vs_predicted.png' using XGBoost
xgb_best = models['XGBoost Regressor']
val_preds = xgb_best.predict(X_val)

importances = pd.Series(
    xgb_best.feature_importances_, index=X_encoded.columns
).sort_values(ascending=False)
fig, ax = plt.subplots(figsize=(8.5, 4.8))
top_feats = importances.head(8)
sns.barplot(
    x=top_feats.values,
    y=top_feats.index,
    hue=top_feats.index,
    palette='Reds_r',
    legend=False,
    ax=ax,
)
ax.set_title(
    'Top 8 Drivers of Delivery Time (XGBoost Regressor)',
    fontsize=13,
    fontweight='bold',
)
ax.set_xlabel('Relative Feature Importance')
ax.set_ylabel('Feature')
plt.tight_layout()
plt.savefig('3_feature_importance.png', dpi=200)
plt.close()

fig, ax = plt.subplots(figsize=(8.5, 4.8))
ax.scatter(y_val, val_preds, alpha=0.35, color='#2C3E50', edgecolors='none')
ax.plot([0, 140], [0, 140], 'r--', lw=2, label='Ideal Prediction (y = x)')
ax.set_title(
    'Actual vs. Predicted Delivery Time (XGBoost Regressor)',
    fontsize=13,
    fontweight='bold',
)
ax.set_xlabel('Actual Delivery Time (Minutes)')
ax.set_ylabel('Predicted Delivery Time (Minutes)')
ax.legend()
plt.tight_layout()
plt.savefig('4_actual_vs_predicted.png', dpi=200)
plt.close()

print(
    'Done! Saved 2_model_comparison.png, 3_feature_importance.png, and'
    ' 4_actual_vs_predicted.png in your folder.'
)