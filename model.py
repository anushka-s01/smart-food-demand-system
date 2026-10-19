"""Prediction service with a trained-model path and transparent fallback."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
import warnings

import pandas as pd

from data_processing import (
    CATEGORICAL_COLUMNS,
    DATA_PATH,
    DAYS,
    FEATURE_COLUMNS,
    MEALS,
    MENUS,
    TARGET_COLUMN,
    YES_NO_VALUES,
    load_and_validate_data,
    prepare_features,
)


MODEL_PATH = Path(__file__).with_name("model.pkl")
RANDOM_SEED = 42


def _validate_choice(name: str, value: str, allowed: tuple[str, ...]) -> None:
    if value not in allowed:
        raise ValueError(f"{name} must be one of: {', '.join(allowed)}")


def _validate_inputs(
    day: str,
    meal: str,
    menu: str,
    holiday: str,
    exam_period: str,
    previous_attendance: int,
) -> int:
    _validate_choice("day", day, DAYS)
    _validate_choice("meal", meal, MEALS)
    _validate_choice("menu", menu, MENUS)
    _validate_choice("holiday", holiday, YES_NO_VALUES)
    _validate_choice("exam_period", exam_period, YES_NO_VALUES)
    try:
        previous = int(previous_attendance)
    except (TypeError, ValueError) as exc:
        raise ValueError("previous_attendance must be an integer from 0 to 5000") from exc
    if not 0 <= previous <= 5000:
        raise ValueError("previous_attendance must be from 0 to 5000")
    return previous


def build_model():
    """Build the reproducible preprocessing and Random Forest pipeline."""
    from sklearn.compose import ColumnTransformer
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import OneHotEncoder

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "categorical",
                OneHotEncoder(handle_unknown="ignore"),
                CATEGORICAL_COLUMNS,
            )
        ],
        remainder="passthrough",
    )
    return Pipeline(
        steps=[
            ("preprocess", preprocessor),
            (
                "regressor",
                RandomForestRegressor(
                    n_estimators=200,
                    min_samples_leaf=2,
                    random_state=RANDOM_SEED,
                    n_jobs=-1,
                ),
            ),
        ]
    )


def train_and_save_model() -> tuple[object, float]:
    """Train on the provided demonstration dataset and persist the pipeline."""
    import joblib
    from sklearn.metrics import mean_absolute_error
    from sklearn.model_selection import train_test_split

    data = load_and_validate_data(DATA_PATH)
    features, target = prepare_features(data)
    x_train, x_test, y_train, y_test = train_test_split(
        features,
        target,
        test_size=0.20,
        random_state=RANDOM_SEED,
    )
    trained_model = build_model()
    trained_model.fit(x_train, y_train)
    mae = float(mean_absolute_error(y_test, trained_model.predict(x_test)))
    joblib.dump(trained_model, MODEL_PATH)
    load_trained_model.cache_clear()
    return trained_model, mae


@lru_cache(maxsize=1)
def load_trained_model():
    """Load the trusted local model, training a replacement if it is absent."""
    import joblib
    from sklearn.pipeline import Pipeline

    if not MODEL_PATH.exists():
        model, _ = train_and_save_model()
        return model

    model = joblib.load(MODEL_PATH)
    if not isinstance(model, Pipeline):
        raise TypeError("model.pkl does not contain the expected sklearn Pipeline")
    if list(getattr(model, "feature_names_in_", [])) != FEATURE_COLUMNS:
        raise ValueError("model.pkl was trained with unexpected feature columns")
    return model


def _historical_fallback(
    day: str,
    meal: str,
    menu: str,
    holiday: str,
    exam_period: str,
    previous: int,
) -> int:
    """Estimate demand from similar rows when sklearn cannot be loaded."""
    try:
        data = load_and_validate_data(DATA_PATH)
    except (FileNotFoundError, ValueError, OSError):
        return previous

    requested = {
        "Day": day,
        "Meal": meal,
        "Menu": menu,
        "Holiday": holiday,
        "Exam_Period": exam_period,
    }
    scores = pd.Series(0, index=data.index, dtype="int64")
    for column, value in requested.items():
        scores += (data[column] == value).astype("int64")

    candidates = data.assign(_score=scores).sort_values("_score", ascending=False).head(30)
    weights = candidates["_score"].clip(lower=1)
    historical_average = float((candidates[TARGET_COLUMN] * weights).sum() / weights.sum())
    estimate = round((0.60 * previous) + (0.40 * historical_average))
    return max(0, int(estimate))


def prediction_engine_status() -> tuple[str, str]:
    """Return the prediction method currently available to the application."""
    try:
        load_trained_model()
    except Exception as exc:
        return "Historical-average fallback", str(exc)
    return "Random Forest model", "Loaded successfully"


def predict_demand(
    day: str,
    meal: str,
    menu: str,
    holiday: str,
    exam_period: str,
    previous_attendance: int,
) -> int:
    """Predict attendance using the trained model or the documented fallback."""
    previous = _validate_inputs(
        day, meal, menu, holiday, exam_period, previous_attendance
    )
    input_row = pd.DataFrame(
        [
            {
                "Day": day,
                "Meal": meal,
                "Menu": menu,
                "Holiday": holiday,
                "Exam_Period": exam_period,
                "Previous_Attendance": previous,
            }
        ],
        columns=FEATURE_COLUMNS,
    )

    try:
        prediction = load_trained_model().predict(input_row)[0]
        return max(0, int(round(float(prediction))))
    except Exception as exc:
        warnings.warn(
            f"Using the historical-average fallback because the trained model "
            f"could not be loaded: {exc}",
            RuntimeWarning,
            stacklevel=2,
        )
        return _historical_fallback(
            day, meal, menu, holiday, exam_period, previous
        )


if __name__ == "__main__":
    _, evaluation_mae = train_and_save_model()
    print(f"Held-out MAE on the synthetic dataset: {evaluation_mae:.2f} students")
    sample = predict_demand("Monday", "Lunch", "Biryani", "No", "No", 500)
    print(f"Sample predicted attendance: {sample}")
