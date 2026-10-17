from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib

from .config import ARTIFACT_DIR, CROP_FEATURES, YIELD_FEATURES
from .data import forecast_yield_features, load_crop_data, load_yield_data
from .models import model_feature_importance, train_classifiers, train_regressors


def train_project(output_dir: Path = ARTIFACT_DIR, random_state: int = 42) -> dict[str, object]:
    output_dir.mkdir(parents=True, exist_ok=True)

    crop_bundle = load_crop_data()
    yield_bundle = load_yield_data()

    crop_result = train_classifiers(crop_bundle.frame, random_state=random_state)
    yield_result = train_regressors(yield_bundle.frame, random_state=random_state)

    crop_model_path = output_dir / "crop_classifier.joblib"
    yield_model_path = output_dir / "yield_regressor.joblib"
    crop_metrics_path = output_dir / "crop_metrics.csv"
    yield_metrics_path = output_dir / "yield_metrics.csv"
    metadata_path = output_dir / "metadata.json"
    crop_importance_path = output_dir / "crop_feature_importance.csv"
    yield_importance_path = output_dir / "yield_feature_importance.csv"
    completed_yield_path = output_dir / "completed_yield_data.csv"
    forecast_path = output_dir / "yield_2025_forecast.csv"

    joblib.dump(crop_result.best_model, crop_model_path)
    joblib.dump(yield_result.best_model, yield_model_path)
    crop_result.metrics.to_csv(crop_metrics_path, index=False)
    yield_result.metrics.to_csv(yield_metrics_path, index=False)

    crop_importance = model_feature_importance(
        crop_result.best_model,
        crop_bundle.frame[CROP_FEATURES],
        crop_bundle.frame["label"],
        scoring="accuracy",
    )
    crop_importance.to_csv(crop_importance_path, index=False)

    completed_yield = yield_result.extra["completed_data"]
    completed_yield.to_csv(completed_yield_path, index=False)
    yield_importance = model_feature_importance(
        yield_result.best_model,
        completed_yield[YIELD_FEATURES],
        completed_yield["hg_per_ha_yield"],
        scoring="r2",
    )
    yield_importance.to_csv(yield_importance_path, index=False)

    forecast_features = forecast_yield_features(yield_bundle.frame, year=2025)
    forecast_features["predicted_hg_per_ha_yield"] = yield_result.best_model.predict(forecast_features)
    forecast_features.to_csv(forecast_path, index=False)

    metadata = {
        "crop_data_source": crop_bundle.source,
        "yield_data_source": yield_bundle.source,
        "best_crop_model": crop_result.best_name,
        "best_yield_model": yield_result.best_name,
        "crop_features": CROP_FEATURES,
        "yield_features": YIELD_FEATURES,
        "artifacts": {
            "crop_model": str(crop_model_path),
            "yield_model": str(yield_model_path),
            "crop_metrics": str(crop_metrics_path),
            "yield_metrics": str(yield_metrics_path),
            "crop_feature_importance": str(crop_importance_path),
            "yield_feature_importance": str(yield_importance_path),
            "completed_yield_data": str(completed_yield_path),
            "yield_2025_forecast": str(forecast_path),
        },
    }
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser(description="Train crop recommendation and wheat yield models.")
    parser.add_argument("--output-dir", type=Path, default=ARTIFACT_DIR)
    parser.add_argument("--random-state", type=int, default=42)
    args = parser.parse_args()

    metadata = train_project(output_dir=args.output_dir, random_state=args.random_state)
    print("Training complete")
    print(f"Best crop model: {metadata['best_crop_model']}")
    print(f"Best yield model: {metadata['best_yield_model']}")
    print(f"Artifacts: {metadata['artifacts']}")


if __name__ == "__main__":
    main()
