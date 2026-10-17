from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from flask import Flask, jsonify, render_template, request

from agri_ml.config import ARTIFACT_DIR, CROP_FEATURES, YIELD_FEATURES
from agri_ml.models import predict_crop
from agri_ml.train import train_project


APP_ROOT = Path(__file__).resolve().parent
app = Flask(
    __name__,
    template_folder=str(APP_ROOT / "web" / "templates"),
    static_folder=str(APP_ROOT / "web" / "static"),
)


CROP_PRESETS = {
    "Balanced field": {"N": 80.0, "P": 45.0, "K": 40.0, "temperature": 25.0, "humidity": 80.0, "ph": 6.5, "rainfall": 120.0},
    "High rainfall lowland": {"N": 78.0, "P": 47.0, "K": 40.0, "temperature": 24.0, "humidity": 84.0, "ph": 6.4, "rainfall": 220.0},
    "Warm dry plot": {"N": 35.0, "P": 55.0, "K": 28.0, "temperature": 31.0, "humidity": 48.0, "ph": 6.7, "rainfall": 58.0},
    "Fruit orchard": {"N": 24.0, "P": 110.0, "K": 190.0, "temperature": 23.0, "humidity": 88.0, "ph": 6.0, "rainfall": 100.0},
    "Commercial cotton": {"N": 118.0, "P": 47.0, "K": 22.0, "temperature": 25.0, "humidity": 78.0, "ph": 6.9, "rainfall": 82.0},
}


CROP_PROFILES = [
    {"name": "Rice", "group": "Cereal", "season": "Monsoon", "water": "High", "duration": "120-150 days", "soil": "Clay loam", "image": "https://images.unsplash.com/photo-1500382017468-9049fed747ef?auto=format&fit=crop&w=1200&q=80"},
    {"name": "Wheat", "group": "Cereal", "season": "Rabi", "water": "Medium", "duration": "110-140 days", "soil": "Well drained loam", "image": "https://images.unsplash.com/photo-1574323347407-f5e1ad6d020b?auto=format&fit=crop&w=1200&q=80"},
    {"name": "Maize", "group": "Cereal", "season": "Kharif", "water": "Medium", "duration": "90-120 days", "soil": "Fertile loam", "image": "https://images.unsplash.com/photo-1601478568631-4305c62bd96c?auto=format&fit=crop&w=1200&q=80"},
    {"name": "Jute", "group": "Fiber", "season": "Monsoon", "water": "High", "duration": "100-120 days", "soil": "Alluvial soil", "image": "https://images.unsplash.com/photo-1560493676-04071c5f467b?auto=format&fit=crop&w=1200&q=80"},
    {"name": "Cotton", "group": "Fiber", "season": "Kharif", "water": "Medium", "duration": "150-180 days", "soil": "Black soil", "image": "https://images.unsplash.com/photo-1598515214211-89d3c73ae83b?auto=format&fit=crop&w=1200&q=80"},
    {"name": "Banana", "group": "Fruit", "season": "Annual", "water": "High", "duration": "10-15 months", "soil": "Deep loam", "image": "https://images.unsplash.com/photo-1528825871115-3581a5387919?auto=format&fit=crop&w=1200&q=80"},
    {"name": "Mango", "group": "Fruit", "season": "Perennial", "water": "Low", "duration": "Perennial", "soil": "Sandy loam", "image": "https://images.unsplash.com/photo-1553279768-865429fa0078?auto=format&fit=crop&w=1200&q=80"},
    {"name": "Coffee", "group": "Plantation", "season": "Perennial", "water": "Medium", "duration": "Perennial", "soil": "Slightly acidic", "image": "https://images.unsplash.com/photo-1525088553748-01d6e210e00b?auto=format&fit=crop&w=1200&q=80"},
]


def ensure_artifacts() -> None:
    required = [
        ARTIFACT_DIR / "metadata.json",
        ARTIFACT_DIR / "crop_classifier.joblib",
        ARTIFACT_DIR / "yield_regressor.joblib",
        ARTIFACT_DIR / "crop_metrics.csv",
        ARTIFACT_DIR / "yield_metrics.csv",
        ARTIFACT_DIR / "crop_feature_importance.csv",
        ARTIFACT_DIR / "yield_feature_importance.csv",
        ARTIFACT_DIR / "completed_yield_data.csv",
        ARTIFACT_DIR / "yield_2025_forecast.csv",
    ]
    if any(not path.exists() for path in required):
        train_project()


def load_state() -> dict[str, Any]:
    ensure_artifacts()
    metadata = json.loads((ARTIFACT_DIR / "metadata.json").read_text(encoding="utf-8"))
    return {
        "metadata": metadata,
        "crop_model": joblib.load(ARTIFACT_DIR / "crop_classifier.joblib"),
        "yield_model": joblib.load(ARTIFACT_DIR / "yield_regressor.joblib"),
        "crop_metrics": pd.read_csv(ARTIFACT_DIR / "crop_metrics.csv"),
        "yield_metrics": pd.read_csv(ARTIFACT_DIR / "yield_metrics.csv"),
        "crop_importance": pd.read_csv(ARTIFACT_DIR / "crop_feature_importance.csv"),
        "yield_importance": pd.read_csv(ARTIFACT_DIR / "yield_feature_importance.csv"),
        "completed_yield": pd.read_csv(ARTIFACT_DIR / "completed_yield_data.csv"),
        "yield_forecast": pd.read_csv(ARTIFACT_DIR / "yield_2025_forecast.csv"),
    }


STATE = load_state()


def json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [json_safe(item) for item in value]
    if isinstance(value, tuple):
        return [json_safe(item) for item in value]
    if isinstance(value, pd.DataFrame):
        return json_safe(value.replace({np.nan: None}).to_dict(orient="records"))
    if isinstance(value, pd.Series):
        return json_safe(value.replace({np.nan: None}).to_dict())
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        if np.isnan(value):
            return None
        return float(value)
    if isinstance(value, float) and np.isnan(value):
        return None
    if isinstance(value, np.ndarray):
        return json_safe(value.tolist())
    return value


def readiness_score(values: dict[str, float]) -> tuple[int, str]:
    ph_score = max(0.0, 1.0 - abs(values["ph"] - 6.5) / 2.8)
    moisture_score = min(1.0, (values["humidity"] / 90.0) * 0.55 + (min(values["rainfall"], 220.0) / 220.0) * 0.45)
    climate_score = max(0.0, 1.0 - abs(values["temperature"] - 26.0) / 18.0)
    nutrients = np.array([values["N"], values["P"], values["K"]], dtype=float)
    nutrient_score = min(1.0, nutrients.mean() / 95.0) * (1.0 - min(0.45, nutrients.std() / 180.0))
    score = int(round(100 * (0.28 * ph_score + 0.28 * moisture_score + 0.22 * climate_score + 0.22 * nutrient_score)))
    band = "Excellent" if score >= 82 else "Good" if score >= 68 else "Watch" if score >= 52 else "Risk"
    return score, band


def recommendation_actions(values: dict[str, float], top_crop: str) -> list[dict[str, str]]:
    return [
        {"area": "Soil pH", "signal": f"{values['ph']:.1f}", "action": "Ready" if 5.8 <= values["ph"] <= 7.4 else "Review pH correction"},
        {"area": "Water", "signal": f"{values['rainfall']:.0f} mm", "action": "Ready" if values["rainfall"] >= 70 else "Plan irrigation"},
        {"area": "Climate", "signal": f"{values['temperature']:.1f} C", "action": "Ready" if 18 <= values["temperature"] <= 34 else "Check seasonal fit"},
        {"area": "Nutrients", "signal": f"NPK {values['N']:.0f}/{values['P']:.0f}/{values['K']:.0f}", "action": f"Validate for {top_crop.title()}"},
    ]


def normalize_crop_payload(payload: dict[str, Any]) -> dict[str, float]:
    return {feature: float(payload.get(feature, CROP_PRESETS["Balanced field"][feature])) for feature in CROP_FEATURES}


def crop_sensitivity(values: dict[str, float], feature: str) -> list[dict[str, Any]]:
    ranges = {
        "rainfall": np.linspace(max(0, values[feature] - 80), values[feature] + 80, 11),
        "temperature": np.linspace(max(5, values[feature] - 7), min(45, values[feature] + 7), 11),
        "humidity": np.linspace(max(10, values[feature] - 25), min(100, values[feature] + 25), 11),
        "ph": np.linspace(max(3.5, values[feature] - 1.2), min(9.5, values[feature] + 1.2), 11),
        "N": np.linspace(max(0, values[feature] - 40), values[feature] + 40, 11),
        "P": np.linspace(max(0, values[feature] - 40), values[feature] + 40, 11),
        "K": np.linspace(max(0, values[feature] - 40), values[feature] + 40, 11),
    }
    rows = []
    for candidate in ranges.get(feature, ranges["rainfall"]):
        scenario = dict(values)
        scenario[feature] = float(candidate)
        prediction = predict_crop(STATE["crop_model"], scenario, top_n=1).iloc[0]
        confidence = prediction["confidence"]
        rows.append(
            {
                "x": round(float(candidate), 2),
                "crop": str(prediction["crop"]).title(),
                "confidence": None if pd.isna(confidence) else round(float(confidence), 4),
            }
        )
    return rows


def yield_stress_table(values: dict[str, float]) -> list[dict[str, Any]]:
    rows = []
    for rainfall_delta in [-20, -10, 0, 10, 20]:
        for temp_delta in [-1.0, 0.0, 1.0]:
            scenario = dict(values)
            scenario["average_rain_fall_mm_per_year"] = max(0.0, values["average_rain_fall_mm_per_year"] * (1 + rainfall_delta / 100))
            scenario["avg_temp"] = values["avg_temp"] + temp_delta
            frame = pd.DataFrame([scenario], columns=YIELD_FEATURES)
            rows.append(
                {
                    "rainfallDelta": rainfall_delta,
                    "tempDelta": temp_delta,
                    "prediction": round(float(STATE["yield_model"].predict(frame)[0]), 2),
                }
            )
    return rows


@app.get("/")
def index():
    return render_template("index.html")


@app.get("/api/state")
def api_state():
    metadata = STATE["metadata"]
    forecast = STATE["yield_forecast"].iloc[0].to_dict()
    return jsonify(
        json_safe(
            {
                "metadata": metadata,
                "cropPresets": CROP_PRESETS,
                "cropProfiles": CROP_PROFILES,
                "cropFeatures": CROP_FEATURES,
                "yieldFeatures": YIELD_FEATURES,
                "cropMetrics": STATE["crop_metrics"],
                "yieldMetrics": STATE["yield_metrics"],
                "cropImportance": STATE["crop_importance"],
                "yieldImportance": STATE["yield_importance"],
                "yieldForecast": forecast,
            }
        )
    )


@app.post("/api/crop")
def api_crop():
    payload = request.get_json(force=True) or {}
    values = normalize_crop_payload(payload)
    stress_feature = str(payload.get("stressFeature", "rainfall"))
    recommendations = predict_crop(STATE["crop_model"], values, top_n=5)
    top_crop = str(recommendations.iloc[0]["crop"]).title()
    score, band = readiness_score(values)
    recs = []
    for row in recommendations.to_dict(orient="records"):
        confidence = row["confidence"]
        recs.append({"crop": str(row["crop"]).title(), "confidence": None if pd.isna(confidence) else round(float(confidence), 4)})
    return jsonify(
        {
            "values": values,
            "recommendations": recs,
            "topCrop": top_crop,
            "readiness": {"score": score, "band": band},
            "actions": recommendation_actions(values, top_crop),
            "sensitivity": crop_sensitivity(values, stress_feature),
        }
    )


@app.post("/api/yield")
def api_yield():
    payload = request.get_json(force=True) or {}
    baseline = STATE["yield_forecast"].iloc[0].to_dict()
    values = {feature: float(payload.get(feature, baseline[feature])) for feature in YIELD_FEATURES}
    frame = pd.DataFrame([values], columns=YIELD_FEATURES)
    prediction = float(STATE["yield_model"].predict(frame)[0])
    baseline_prediction = float(baseline["predicted_hg_per_ha_yield"])
    return jsonify(
        {
            "values": values,
            "prediction": round(prediction, 2),
            "baseline": round(baseline_prediction, 2),
            "delta": round(prediction - baseline_prediction, 2),
            "stress": yield_stress_table(values),
        }
    )


@app.post("/api/field-plan")
def api_field_plan():
    payload = request.get_json(force=True) or {}
    fields = payload.get("fields") or []
    rows = []
    for item in fields:
        scenario = str(item.get("scenario", "Balanced field"))
        acres = int(item.get("acres", 5))
        values = CROP_PRESETS.get(scenario, CROP_PRESETS["Balanced field"])
        prediction = predict_crop(STATE["crop_model"], values, top_n=1).iloc[0]
        score, band = readiness_score(values)
        rows.append(
            {
                "field": str(item.get("field", "Field")),
                "scenario": scenario,
                "acres": acres,
                "crop": str(prediction["crop"]).title(),
                "readiness": score,
                "band": band,
            }
        )
    plan = pd.DataFrame(rows)
    mix = plan.groupby("crop", as_index=False)["acres"].sum().sort_values("acres", ascending=False).to_dict(orient="records") if not plan.empty else []
    return jsonify({"fields": rows, "mix": mix})


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=False)
