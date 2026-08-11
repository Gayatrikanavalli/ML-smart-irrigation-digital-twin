"""
Trains two models on data/irrigation_data.csv:
1. Classifier: predicts whether irrigation is needed today (needs_irrigation)
2. Regressor: forecasts next day's soil moisture (next_day_moisture)

These two models together are the "Hybrid ML" part of the project -
a classification model for decisions + a regression/forecasting model for
predicting future field state (which feeds the Digital Twin simulation).

Run: python3 train_model.py
Output: models/classifier.pkl, models/regressor.pkl
"""

import pandas as pd 
import numpy as np
import joblib
import os

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    accuracy_score, f1_score, classification_report,
    mean_absolute_error, mean_squared_error, r2_score
)

df = pd.read_csv("data/irrigation_data.csv")

# Encode crop_type as a number for the model
le = LabelEncoder()
df["crop_type_encoded"] = le.fit_transform(df["crop_type"])

FEATURES = [
    "crop_type_encoded", "crop_day", "temperature_c",
    "humidity_pct", "rainfall_mm", "soil_moisture_pct", "et_loss"
]

# ---------- 1. Classifier: needs_irrigation ----------
X = df[FEATURES]
y_clf = df["needs_irrigation"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y_clf, test_size=0.2, random_state=42, stratify=y_clf
)

clf = RandomForestClassifier(n_estimators=200, max_depth=8, random_state=42, class_weight="balanced")
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)
print("=== CLASSIFIER: Irrigation Decision ===")
print(f"Accuracy: {accuracy_score(y_test, y_pred):.3f}")
print(f"F1 Score: {f1_score(y_test, y_pred):.3f}")
print(classification_report(y_test, y_pred))

# ---------- 2. Regressor: next_day_moisture ----------
y_reg = df["next_day_moisture"]

X_train2, X_test2, y_train2, y_test2 = train_test_split(
    X, y_reg, test_size=0.2, random_state=42
)

reg = RandomForestRegressor(n_estimators=200, max_depth=10, random_state=42)
reg.fit(X_train2, y_train2)

y_pred2 = reg.predict(X_test2)
print("\n=== REGRESSOR: Next-Day Soil Moisture Forecast ===")
print(f"MAE:  {mean_absolute_error(y_test2, y_pred2):.3f}")
print(f"RMSE: {np.sqrt(mean_squared_error(y_test2, y_pred2)):.3f}")
print(f"R2:   {r2_score(y_test2, y_pred2):.3f}")

# ---------- Save everything ----------
os.makedirs("models", exist_ok=True)
joblib.dump(clf, "models/classifier.pkl")
joblib.dump(reg, "models/regressor.pkl")
joblib.dump(le, "models/label_encoder.pkl")
joblib.dump(FEATURES, "models/feature_list.pkl")

print("\nModels saved to models/ folder:")
print(" - classifier.pkl   (irrigation decision)")
print(" - regressor.pkl    (moisture forecast)")
print(" - label_encoder.pkl")
print(" - feature_list.pkl")
