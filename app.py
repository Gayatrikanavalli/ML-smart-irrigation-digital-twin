"""
Backend for the Smart Irrigation Digital Twin project.
Run:  python app.py
Then open http://localhost:8000/docs to test every endpoint.
"""

import random
import numpy as np
import joblib
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

clf = joblib.load("models/classifier.pkl")
reg = joblib.load("models/regressor.pkl")
le = joblib.load("models/label_encoder.pkl")
FEATURES = joblib.load("models/feature_list.pkl")

CROP_PROFILES = {
    "Wheat":     {"ideal_moisture": 45, "et_rate": 1.8, "growth_days": 120},
    "Rice":      {"ideal_moisture": 70, "et_rate": 2.5, "growth_days": 130},
    "Maize":     {"ideal_moisture": 50, "et_rate": 2.0, "growth_days": 100},
    "Cotton":    {"ideal_moisture": 40, "et_rate": 1.7, "growth_days": 150},
    "Sugarcane": {"ideal_moisture": 60, "et_rate": 2.2, "growth_days": 300},
}
KNOWN_CROPS = list(le.classes_)

app = FastAPI(title="Smart Irrigation Digital Twin API")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

NUM_PLOTS = 6
field_state = {}
history = []
baseline_water_used = 0.0
model_water_used = 0.0
sim_day_counter = 0


def encode_crop(crop_name):
    if crop_name in KNOWN_CROPS:
        return int(le.transform([crop_name])[0])
    return 0


def make_new_plot(plot_id):
    crop = random.choice(KNOWN_CROPS)
    profile = CROP_PROFILES[crop]
    moisture_val = profile["ideal_moisture"] + random.uniform(-8, 8)
    return {
        "plot_id": plot_id,
        "crop_type": crop,
        "crop_day": random.randint(0, 30),
        "temperature_c": round(random.uniform(22, 34), 1),
        "humidity_pct": round(random.uniform(35, 75), 1),
        "rainfall_mm": 0.0,
        "soil_moisture_pct": round(moisture_val, 2),
        "et_loss": profile["et_rate"],
        "last_irrigated_day": -1,
    }


def reset_field():
    global field_state, history, baseline_water_used, model_water_used, sim_day_counter
    field_state = {i: make_new_plot(i) for i in range(1, NUM_PLOTS + 1)}
    history = []
    baseline_water_used = 0.0
    model_water_used = 0.0
    sim_day_counter = 0


reset_field()


def build_feature_row(plot):
    row = {
        "crop_type_encoded": encode_crop(plot["crop_type"]),
        "crop_day": plot["crop_day"],
        "temperature_c": plot["temperature_c"],
        "humidity_pct": plot["humidity_pct"],
        "rainfall_mm": plot["rainfall_mm"],
        "soil_moisture_pct": plot["soil_moisture_pct"],
        "et_loss": plot["et_loss"],
    }
    return [row[f] for f in FEATURES]


def predict_for_plot(plot):
    x = [build_feature_row(plot)]
    needs_irrigation = int(clf.predict(x)[0])
    next_moisture = float(reg.predict(x)[0])
    profile = CROP_PROFILES[plot["crop_type"]]
    deficit = max(0, profile["ideal_moisture"] - plot["soil_moisture_pct"])
    water_amount = round(deficit * 1.2, 2) if needs_irrigation else 0.0
    return {
        "needs_irrigation": bool(needs_irrigation),
        "predicted_next_day_moisture": round(next_moisture, 2),
        "recommended_water_mm": water_amount,
    }


@app.get("/field-state")
def get_field_state():
    return {"plots": list(field_state.values())}


@app.post("/predict/{plot_id}")
def predict(plot_id: int):
    if plot_id not in field_state:
        raise HTTPException(404, "Plot not found")
    plot = field_state[plot_id]
    result = predict_for_plot(plot)
    return {"plot_id": plot_id, **result}


@app.post("/simulate/{days}")
def simulate(days: int):
    global baseline_water_used, model_water_used, sim_day_counter

    if days < 1 or days > 90:
        raise HTTPException(400, "days must be between 1 and 90")

    for _ in range(days):
        sim_day_counter += 1
        for plot_id, plot in field_state.items():
            profile = CROP_PROFILES[plot["crop_type"]]

            plot["temperature_c"] = round(np.clip(np.random.normal(28, 5), 10, 45), 2)
            plot["humidity_pct"] = round(np.clip(np.random.normal(55, 15), 10, 95), 2)
            plot["rainfall_mm"] = round(max(0, np.random.exponential(2) - 1.5), 2)

            temp_factor = 1 + (plot["temperature_c"] - 28) / 40
            humidity_factor = 1 - plot["humidity_pct"] / 200
            raw_et = profile["et_rate"] * temp_factor * humidity_factor
            plot["et_loss"] = round(max(0.2, raw_et), 2)
            plot["crop_day"] = (plot["crop_day"] + 1) % profile["growth_days"]

            new_moisture = plot["soil_moisture_pct"] - plot["et_loss"] + plot["rainfall_mm"] * 0.8
            plot["soil_moisture_pct"] = float(np.clip(new_moisture, 0, 100))

            pred = predict_for_plot(plot)
            if pred["needs_irrigation"]:
                plot["soil_moisture_pct"] = min(100, plot["soil_moisture_pct"] + pred["recommended_water_mm"])
                plot["last_irrigated_day"] = sim_day_counter
                model_water_used += pred["recommended_water_mm"]

            if sim_day_counter % 5 == 0:
                baseline_water_used += 10.0

            history.append({
                "day": sim_day_counter,
                "plot_id": plot_id,
                "crop_type": plot["crop_type"],
                "soil_moisture_pct": round(plot["soil_moisture_pct"], 2),
                "irrigated": pred["needs_irrigation"],
                "water_applied_mm": pred["recommended_water_mm"],
            })

    water_saved_pct = 0.0
    if baseline_water_used > 0:
        saved = baseline_water_used - model_water_used
        water_saved_pct = round(100 * saved / baseline_water_used, 1)

    return {
        "message": f"Simulated {days} day(s).",
        "current_day": sim_day_counter,
        "model_water_used_mm": round(model_water_used, 2),
        "baseline_water_used_mm": round(baseline_water_used, 2),
        "water_saved_pct": water_saved_pct,
        "field_state": list(field_state.values()),
    }


@app.get("/simulate/history")
def get_history():
    return {"history": history}


@app.post("/reset")
def reset():
    reset_field()
    return {"message": "Field reset.", "field_state": list(field_state.values())}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)