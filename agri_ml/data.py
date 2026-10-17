from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import IterativeImputer

from .config import (
    CROP_DATA_FILE,
    CROP_FEATURES,
    CROP_LABELS,
    CROP_TARGET,
    YIELD_DATA_FILE,
    YIELD_FEATURES,
    YIELD_TARGET,
)


@dataclass(frozen=True)
class DatasetBundle:
    frame: pd.DataFrame
    source: str


def load_crop_data(path: Path | str = CROP_DATA_FILE) -> DatasetBundle:
    csv_path = Path(path)
    if csv_path.exists():
        frame = _normalize_crop_columns(pd.read_csv(csv_path))
        return DatasetBundle(frame=frame, source=str(csv_path))
    return DatasetBundle(frame=make_sample_crop_data(), source="generated sample data")


def load_yield_data(path: Path | str = YIELD_DATA_FILE) -> DatasetBundle:
    csv_path = Path(path)
    if csv_path.exists():
        frame = _normalize_yield_columns(pd.read_csv(csv_path))
        return DatasetBundle(frame=frame, source=str(csv_path))
    return DatasetBundle(frame=make_sample_yield_data(), source="generated sample data")


def make_sample_crop_data(seed: int = 42, samples_per_crop: int = 100) -> pd.DataFrame:
    """Generate a deterministic crop dataset with the same schema as Kaggle's crop file."""
    rng = np.random.default_rng(seed)
    prototypes = _crop_prototypes()
    rows: list[dict[str, float | str]] = []

    for label in CROP_LABELS:
        base = prototypes[label]
        for _ in range(samples_per_crop):
            row = {
                feature: float(max(0.0, rng.normal(mean, spread)))
                for feature, (mean, spread) in base.items()
            }
            row["ph"] = float(np.clip(row["ph"], 3.5, 9.5))
            row["temperature"] = float(np.clip(row["temperature"], 5.0, 45.0))
            row["humidity"] = float(np.clip(row["humidity"], 10.0, 100.0))
            row["label"] = label
            rows.append(row)

    return pd.DataFrame(rows, columns=CROP_FEATURES + [CROP_TARGET])


def make_sample_yield_data(seed: int = 7) -> pd.DataFrame:
    """Create Pakistan wheat-like yearly data with missing future yield values for MICE."""
    rng = np.random.default_rng(seed)
    rows: list[dict[str, float | int | str]] = []

    for year in range(1992, 2025):
        rainfall = 345 + 3.4 * (year - 1992) + rng.normal(0, 22)
        pesticides = 18200 + 760 * (year - 1992) + rng.normal(0, 900)
        avg_temp = 20.1 + 0.035 * (year - 1992) + rng.normal(0, 0.18)
        yield_value = (
            19000
            + 205 * (year - 1992)
            + 8.0 * rainfall
            - 0.018 * pesticides
            - 135 * avg_temp
            + rng.normal(0, 180)
        )
        rows.append(
            {
                "Area": "Pakistan",
                "Item": "Wheat",
                "Year": year,
                "average_rain_fall_mm_per_year": max(0, rainfall),
                "pesticides_tonnes": max(0, pesticides),
                "avg_temp": avg_temp,
                YIELD_TARGET: yield_value if year <= 2013 else np.nan,
            }
        )

    return pd.DataFrame(rows)


def complete_yield_with_mice(frame: pd.DataFrame, random_state: int = 42) -> pd.DataFrame:
    work = frame.copy()
    required = YIELD_FEATURES + [YIELD_TARGET]
    missing = [column for column in required if column not in work.columns]
    if missing:
        raise ValueError(f"Yield data is missing required columns: {missing}")

    work["yield_was_observed"] = work[YIELD_TARGET].notna()
    imputer = IterativeImputer(max_iter=20, random_state=random_state, sample_posterior=False)
    imputed = imputer.fit_transform(work[required])
    completed = work.copy()
    completed[required] = imputed
    completed["Year"] = completed["Year"].round().astype(int)
    return completed


def forecast_yield_features(frame: pd.DataFrame, year: int = 2025) -> pd.DataFrame:
    """Estimate future weather/input features from simple historical trends."""
    completed = complete_yield_with_mice(frame)
    feature_values: dict[str, float | int] = {"Year": year}
    years = completed[["Year"]].to_numpy()

    for feature in YIELD_FEATURES:
        if feature == "Year":
            continue
        model = LinearRegression()
        model.fit(years, completed[feature].to_numpy())
        feature_values[feature] = float(model.predict(np.array([[year]]))[0])

    return pd.DataFrame([feature_values], columns=YIELD_FEATURES)


def _normalize_crop_columns(frame: pd.DataFrame) -> pd.DataFrame:
    aliases = {
        "n": "N",
        "nitrogen": "N",
        "p": "P",
        "phosphorus": "P",
        "k": "K",
        "potassium": "K",
        "temp": "temperature",
        "temperature": "temperature",
        "humidity": "humidity",
        "ph": "ph",
        "rainfall": "rainfall",
        "rain": "rainfall",
        "label": "label",
        "crop": "label",
        "crop_name": "label",
    }
    normalized = frame.rename(columns={column: aliases.get(_clean_name(column), column) for column in frame.columns})
    missing = [column for column in CROP_FEATURES + [CROP_TARGET] if column not in normalized.columns]
    if missing:
        raise ValueError(f"Crop data is missing required columns: {missing}")
    normalized = normalized[CROP_FEATURES + [CROP_TARGET]].copy()
    normalized[CROP_FEATURES] = normalized[CROP_FEATURES].apply(pd.to_numeric, errors="coerce")
    normalized[CROP_TARGET] = normalized[CROP_TARGET].astype(str).str.strip().str.lower()
    return normalized.dropna().reset_index(drop=True)


def _normalize_yield_columns(frame: pd.DataFrame) -> pd.DataFrame:
    aliases = {
        "year": "Year",
        "average_rain_fall_mm_per_year": "average_rain_fall_mm_per_year",
        "average rainfall": "average_rain_fall_mm_per_year",
        "rainfall": "average_rain_fall_mm_per_year",
        "pesticides_tonnes": "pesticides_tonnes",
        "pesticides": "pesticides_tonnes",
        "avg_temp": "avg_temp",
        "temperature": "avg_temp",
        "hg/ha_yield": YIELD_TARGET,
        "hg_per_ha_yield": YIELD_TARGET,
        "yield": YIELD_TARGET,
        "area": "Area",
        "item": "Item",
    }
    normalized = frame.rename(columns={column: aliases.get(_clean_name(column), column) for column in frame.columns})

    if {"Area", "Item"}.issubset(normalized.columns):
        mask = (
            normalized["Area"].astype(str).str.lower().eq("pakistan")
            & normalized["Item"].astype(str).str.lower().str.contains("wheat", na=False)
        )
        filtered = normalized.loc[mask].copy()
        if not filtered.empty:
            normalized = filtered

    missing = [column for column in YIELD_FEATURES + [YIELD_TARGET] if column not in normalized.columns]
    if missing:
        raise ValueError(f"Yield data is missing required columns: {missing}")

    keep_columns = [column for column in ["Area", "Item"] if column in normalized.columns] + YIELD_FEATURES + [YIELD_TARGET]
    normalized = normalized[keep_columns].copy()
    for column in YIELD_FEATURES + [YIELD_TARGET]:
        normalized[column] = pd.to_numeric(normalized[column], errors="coerce")
    return normalized.dropna(subset=YIELD_FEATURES).sort_values("Year").reset_index(drop=True)


def _clean_name(name: object) -> str:
    return str(name).strip().lower().replace("-", "_").replace(" ", "_")


def _crop_prototypes() -> dict[str, dict[str, tuple[float, float]]]:
    return {
        "rice": _p(78, 47, 40, 24, 82, 6.5, 230),
        "maize": _p(80, 48, 20, 22, 65, 6.3, 85),
        "chickpea": _p(38, 68, 79, 18, 18, 7.2, 80),
        "kidneybeans": _p(22, 67, 20, 20, 22, 5.8, 105),
        "pigeonpeas": _p(20, 68, 20, 28, 48, 5.8, 150),
        "mothbeans": _p(22, 48, 20, 28, 53, 6.8, 52),
        "mungbean": _p(20, 47, 20, 28, 86, 6.7, 48),
        "blackgram": _p(40, 67, 20, 30, 65, 7.0, 68),
        "lentil": _p(20, 68, 20, 24, 64, 6.9, 45),
        "pomegranate": _p(20, 18, 40, 22, 90, 6.4, 107),
        "banana": _p(100, 80, 50, 27, 80, 6.0, 105),
        "mango": _p(20, 28, 30, 31, 50, 5.8, 95),
        "grapes": _p(23, 132, 200, 24, 82, 6.0, 70),
        "watermelon": _p(100, 18, 50, 26, 85, 6.5, 55),
        "muskmelon": _p(100, 18, 50, 28, 92, 6.4, 25),
        "apple": _p(20, 125, 200, 22, 92, 5.9, 112),
        "orange": _p(20, 18, 10, 24, 92, 7.0, 110),
        "papaya": _p(50, 60, 50, 33, 92, 6.7, 145),
        "coconut": _p(20, 18, 30, 27, 95, 5.9, 175),
        "cotton": _p(118, 47, 20, 24, 80, 6.9, 80),
        "jute": _p(78, 47, 40, 25, 79, 6.7, 180),
        "coffee": _p(100, 28, 30, 26, 58, 6.8, 160),
    }


def _p(n: float, p: float, k: float, temp: float, humidity: float, ph: float, rainfall: float) -> dict[str, tuple[float, float]]:
    return {
        "N": (n, 8),
        "P": (p, 7),
        "K": (k, 8),
        "temperature": (temp, 2.2),
        "humidity": (humidity, 5),
        "ph": (ph, 0.35),
        "rainfall": (rainfall, 14),
    }
