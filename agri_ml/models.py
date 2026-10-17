from __future__ import annotations

from dataclasses import dataclass
from math import sqrt
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.ensemble import ExtraTreesClassifier, GradientBoostingRegressor, RandomForestClassifier, RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
)
from sklearn.model_selection import KFold, StratifiedKFold, cross_val_score, train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.svm import SVC, SVR
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

from .config import CROP_FEATURES, CROP_TARGET, YIELD_FEATURES, YIELD_TARGET
from .data import complete_yield_with_mice

try:
    from xgboost import XGBRegressor
except Exception:  # pragma: no cover - xgboost is optional at runtime
    XGBRegressor = None


@dataclass
class TrainingResult:
    best_name: str
    best_model: BaseEstimator
    metrics: pd.DataFrame
    extra: dict[str, Any]


def classifier_candidates(random_state: int = 42) -> dict[str, BaseEstimator]:
    return {
        "Extra Trees": ExtraTreesClassifier(n_estimators=250, random_state=random_state, n_jobs=-1),
        "Logistic Regression": Pipeline(
            [
                ("scale", MinMaxScaler()),
                ("model", LogisticRegression(max_iter=2000, solver="lbfgs")),
            ]
        ),
        "Decision Tree": DecisionTreeClassifier(random_state=random_state),
        "Random Forest": RandomForestClassifier(n_estimators=250, random_state=random_state, n_jobs=-1),
        "K-Nearest Neighbors": Pipeline([("scale", MinMaxScaler()), ("model", KNeighborsClassifier(n_neighbors=5))]),
        "Gaussian Naive Bayes": GaussianNB(),
        "Support Vector Machine": Pipeline(
            [
                ("scale", MinMaxScaler()),
                ("model", SVC(kernel="rbf", C=8.0, gamma="scale", random_state=random_state)),
            ]
        ),
    }


def regressor_candidates(random_state: int = 42) -> dict[str, BaseEstimator]:
    candidates: dict[str, BaseEstimator] = {
        "Decision Tree Regressor": DecisionTreeRegressor(max_depth=6, random_state=random_state),
        "Random Forest Regressor": RandomForestRegressor(n_estimators=300, random_state=random_state, n_jobs=-1),
        "K-Nearest Neighbors Regressor": Pipeline(
            [("scale", StandardScaler()), ("model", KNeighborsRegressor(n_neighbors=3, weights="distance"))]
        ),
        "Support Vector Regressor": Pipeline(
            [("scale", StandardScaler()), ("model", SVR(kernel="rbf", C=100.0, epsilon=0.1, gamma="scale"))]
        ),
    }
    if XGBRegressor is not None:
        candidates["XGBoost Regressor"] = XGBRegressor(
            n_estimators=180,
            max_depth=3,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.9,
            objective="reg:squarederror",
            random_state=random_state,
        )
    else:
        candidates["Gradient Boosting Regressor"] = GradientBoostingRegressor(random_state=random_state)
    return candidates


def train_classifiers(frame: pd.DataFrame, random_state: int = 42) -> TrainingResult:
    clean = frame.dropna(subset=CROP_FEATURES + [CROP_TARGET]).copy()
    x = clean[CROP_FEATURES]
    y = clean[CROP_TARGET].astype(str)
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.2, random_state=random_state, stratify=y
    )

    rows: list[dict[str, float | str]] = []
    fitted: dict[str, BaseEstimator] = {}
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)

    for name, model in classifier_candidates(random_state).items():
        model.fit(x_train, y_train)
        predictions = model.predict(x_test)
        cv_scores = cross_val_score(model, x, y, cv=cv, scoring="accuracy", n_jobs=None)
        rows.append(
            {
                "model": name,
                "accuracy": accuracy_score(y_test, predictions),
                "precision": precision_score(y_test, predictions, average="weighted", zero_division=0),
                "recall": recall_score(y_test, predictions, average="weighted", zero_division=0),
                "f1": f1_score(y_test, predictions, average="weighted", zero_division=0),
                "cv_accuracy_mean": float(cv_scores.mean()),
                "cv_accuracy_std": float(cv_scores.std()),
            }
        )
        fitted[name] = model

    metrics = pd.DataFrame(rows).sort_values(["accuracy", "f1", "cv_accuracy_mean"], ascending=False)
    best_name = str(metrics.iloc[0]["model"])
    best_model = fitted[best_name]
    labels = sorted(y.unique())
    matrix = confusion_matrix(y_test, best_model.predict(x_test), labels=labels)

    return TrainingResult(
        best_name=best_name,
        best_model=best_model,
        metrics=metrics.reset_index(drop=True),
        extra={"labels": labels, "confusion_matrix": matrix, "test_size": len(x_test)},
    )


def train_regressors(frame: pd.DataFrame, random_state: int = 42) -> TrainingResult:
    completed = complete_yield_with_mice(frame, random_state=random_state)
    observed = completed[completed["yield_was_observed"]].copy()
    if len(observed) < 8:
        raise ValueError("At least 8 observed yield rows are required to train regression models.")

    x = observed[YIELD_FEATURES]
    y = observed[YIELD_TARGET]
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=random_state)
    cv_splits = min(5, len(x_train))
    cv = KFold(n_splits=cv_splits, shuffle=True, random_state=random_state)

    rows: list[dict[str, float | str]] = []
    fitted: dict[str, BaseEstimator] = {}

    for name, model in regressor_candidates(random_state).items():
        model.fit(x_train, y_train)
        predictions = model.predict(x_test)
        errors = np.abs(y_test.to_numpy() - predictions)
        cv_scores = cross_val_score(model, x, y, cv=cv, scoring="r2", n_jobs=None)
        rows.append(
            {
                "model": name,
                "r2": r2_score(y_test, predictions),
                "mae": mean_absolute_error(y_test, predictions),
                "mse": mean_squared_error(y_test, predictions),
                "rmse": sqrt(mean_squared_error(y_test, predictions)),
                "std_abs_error": float(errors.std()),
                "cv_r2_mean": float(cv_scores.mean()),
                "cv_r2_std": float(cv_scores.std()),
            }
        )
        fitted[name] = model

    metrics = pd.DataFrame(rows).sort_values(["r2", "rmse"], ascending=[False, True])
    best_name = str(metrics.iloc[0]["model"])
    best_model = fitted[best_name]
    best_model.fit(completed[YIELD_FEATURES], completed[YIELD_TARGET])

    return TrainingResult(
        best_name=best_name,
        best_model=best_model,
        metrics=metrics.reset_index(drop=True),
        extra={"completed_data": completed},
    )


def predict_crop(model: BaseEstimator, values: dict[str, float], top_n: int = 3) -> pd.DataFrame:
    row = pd.DataFrame([{feature: values[feature] for feature in CROP_FEATURES}])
    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(row)[0]
        classes = getattr(model, "classes_", None)
        if classes is None and isinstance(model, Pipeline):
            classes = model.named_steps["model"].classes_
        ranking = sorted(zip(classes, probabilities), key=lambda item: item[1], reverse=True)[:top_n]
        return pd.DataFrame(ranking, columns=["crop", "confidence"])

    prediction = model.predict(row)[0]
    return pd.DataFrame([{"crop": prediction, "confidence": np.nan}])


def model_feature_importance(model: BaseEstimator, x: pd.DataFrame, y: pd.Series | np.ndarray, scoring: str) -> pd.DataFrame:
    estimator = _final_estimator(model)
    values: np.ndarray | None = None

    if hasattr(estimator, "feature_importances_"):
        values = np.asarray(estimator.feature_importances_, dtype=float)
    elif hasattr(estimator, "coef_"):
        coef = np.asarray(estimator.coef_, dtype=float)
        values = np.abs(coef).mean(axis=0) if coef.ndim > 1 else np.abs(coef)

    if values is None or len(values) != x.shape[1]:
        result = permutation_importance(model, x, y, n_repeats=8, random_state=42, scoring=scoring)
        values = result.importances_mean

    values = np.maximum(values, 0)
    if values.sum() > 0:
        values = values / values.sum()
    return (
        pd.DataFrame({"feature": list(x.columns), "importance": values})
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )


def save_training_result(result: TrainingResult, model_path: str | Any, metrics_path: str | Any) -> None:
    joblib.dump(result.best_model, model_path)
    result.metrics.to_csv(metrics_path, index=False)


def _final_estimator(model: BaseEstimator) -> BaseEstimator:
    if isinstance(model, Pipeline):
        return model.steps[-1][1]
    return model
