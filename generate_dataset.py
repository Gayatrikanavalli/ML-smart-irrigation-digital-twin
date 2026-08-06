"""
Generates a realistic synthetic dataset for the Smart Irrigation project.
No Kaggle account needed - this creates data/irrigation_data.csv directly.

Simulates 5 crops across a growing season, with soil moisture that:
- decays over time due to evapotranspiration (affected by temperature/humidity)
- rises sharply when irrigation happens
- has realistic day-to-day weather noise

Run: python3 generate_dataset.py
Output: data/irrigation_data.csv
"""

import numpy as np
import pandas as pd
import os

np.random.seed(42)

CROPS = ["Wheat", "Rice", "Maize", "Cotton", "Sugarcane"]
# Different crops have different ideal moisture ranges and water needs
CROP_PROFILES = {
    "Wheat":     {"ideal_moisture": 45, "et_rate": 1.8, "growth_days": 120},
    "Rice":      {"ideal_moisture": 70, "et_rate": 2.5, "growth_days": 130},
    "Maize":     {"ideal_moisture": 50, "et_rate": 2.0, "growth_days": 100},
    "Cotton":    {"ideal_moisture": 40, "et_rate": 1.7, "growth_days": 150},
    "Sugarcane": {"ideal_moisture": 60, "et_rate": 2.2, "growth_days": 300},
}

NUM_PLOTS = 5          # like 5 fields/plots, each growing a crop
DAYS = 200              # days of history per plot

rows = []

for plot_id in range(1, NUM_PLOTS + 1):
    crop = np.random.choice(CROPS)
    profile = CROP_PROFILES[crop]
    ideal_moisture = profile["ideal_moisture"]
    base_et = profile["et_rate"]

    moisture = ideal_moisture + np.random.uniform(-5, 5)  # starting moisture
    crop_day = 0

    for day in range(DAYS):
        crop_day = (crop_day + 1) % profile["growth_days"]

        # Simulate daily weather
        temperature = np.clip(np.random.normal(28, 5), 10, 45)          # Celsius
        humidity = np.clip(np.random.normal(55, 15), 10, 95)            # %
        rainfall = max(0, np.random.exponential(2) - 1.5)               # mm, mostly 0

        # Evapotranspiration increases with temperature, decreases with humidity
        et_loss = base_et * (1 + (temperature - 28) / 40) * (1 - humidity / 200)
        et_loss = max(0.2, et_loss)

        # Moisture drops due to ET, rises due to rainfall
        moisture = moisture - et_loss + rainfall * 0.8
        moisture = np.clip(moisture, 0, 100)

        # Decision label: irrigate if moisture drops meaningfully below ideal
        needs_irrigation = 1 if moisture < (ideal_moisture - 4) else 0

        # Recommended water amount (mm) - proportional to the deficit
        deficit = max(0, (ideal_moisture - moisture))
        water_amount = round(deficit * 1.2, 2) if needs_irrigation else 0.0

        rows.append({
            "plot_id": plot_id,
            "day": day,
            "crop_type": crop,
            "crop_day": crop_day,
            "temperature_c": round(temperature, 2),
            "humidity_pct": round(humidity, 2),
            "rainfall_mm": round(rainfall, 2),
            "soil_moisture_pct": round(moisture, 2),
            "et_loss": round(et_loss, 2),
            "needs_irrigation": needs_irrigation,
            "water_amount_mm": water_amount,
        })

        # If irrigated, moisture rises back up (simulating the water applied)
        if needs_irrigation:
            moisture = min(100, moisture + water_amount)

df = pd.DataFrame(rows)

# Add a "next day moisture" column - this is the target for the forecasting model
df["next_day_moisture"] = df.groupby("plot_id")["soil_moisture_pct"].shift(-1)
df = df.dropna(subset=["next_day_moisture"])

os.makedirs("data", exist_ok=True)
df.to_csv("data/irrigation_data.csv", index=False)

print(f"Dataset created: data/irrigation_data.csv")
print(f"Shape: {df.shape}")
print(f"\nCrops used: {df['crop_type'].unique().tolist()}")
print(f"\nIrrigation needed distribution:\n{df['needs_irrigation'].value_counts()}")
print(f"\nSample rows:\n{df.head()}")
