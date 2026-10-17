from agri_ml.config import CROP_FEATURES, YIELD_FEATURES
from agri_ml.data import complete_yield_with_mice, forecast_yield_features, make_sample_crop_data, make_sample_yield_data
from agri_ml.models import predict_crop, train_classifiers, train_regressors


def test_sample_data_shapes():
    crop = make_sample_crop_data(samples_per_crop=5)
    yield_data = make_sample_yield_data()

    assert set(CROP_FEATURES + ["label"]).issubset(crop.columns)
    assert set(YIELD_FEATURES + ["hg_per_ha_yield"]).issubset(yield_data.columns)
    assert crop["label"].nunique() == 22
    assert yield_data["hg_per_ha_yield"].isna().sum() > 0


def test_mice_completion_and_forecast_features():
    yield_data = make_sample_yield_data()
    completed = complete_yield_with_mice(yield_data)
    forecast = forecast_yield_features(yield_data, year=2025)

    assert completed["hg_per_ha_yield"].isna().sum() == 0
    assert int(forecast["Year"].iloc[0]) == 2025


def test_training_smoke():
    crop = make_sample_crop_data(samples_per_crop=12)
    yield_data = make_sample_yield_data()

    crop_result = train_classifiers(crop)
    yield_result = train_regressors(yield_data)
    prediction = predict_crop(crop_result.best_model, {feature: float(crop[feature].mean()) for feature in CROP_FEATURES})

    assert not crop_result.metrics.empty
    assert not yield_result.metrics.empty
    assert not prediction.empty
