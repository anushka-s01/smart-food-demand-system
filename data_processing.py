"""Dataset validation and feature preparation."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


DATA_PATH = Path(__file__).with_name("dataset.csv")

FEATURE_COLUMNS = [
    "Day",
    "Meal",
    "Menu",
    "Holiday",
    "Exam_Period",
    "Previous_Attendance",
]
TARGET_COLUMN = "Actual_Attendance"
REQUIRED_COLUMNS = [
    "Date",
    *FEATURE_COLUMNS,
    TARGET_COLUMN,
    "Food_Prepared",
    "Food_Wasted",
]
CATEGORICAL_COLUMNS = ["Day", "Meal", "Menu", "Holiday", "Exam_Period"]

DAYS = (
    "Monday",
    "Tuesday",
    "Wednesday",
    "Thursday",
    "Friday",
    "Saturday",
    "Sunday",
)
MEALS = ("Breakfast", "Lunch", "Dinner")
MENUS = (
    "Aloo Paratha",
    "Biryani",
    "Chole Rice",
    "Dal Khichdi",
    "Dal Rice",
    "Idli Sambar",
    "Paneer Roti",
    "Poha",
    "Rajma Chawal",
    "Roti Sabzi",
    "Upma",
    "Veg Pulao",
)
YES_NO_VALUES = ("No", "Yes")


def load_and_validate_data(path: Path | str = DATA_PATH) -> pd.DataFrame:
    """Load the CSV and raise a useful error if its contract is invalid."""
    dataset_path = Path(path)
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")

    data = pd.read_csv(dataset_path)
    missing = [column for column in REQUIRED_COLUMNS if column not in data.columns]
    if missing:
        raise ValueError(f"dataset.csv is missing required columns: {missing}")
    if data.empty:
        raise ValueError("dataset.csv contains no records")

    data = data[REQUIRED_COLUMNS].copy()
    data["Date"] = pd.to_datetime(data["Date"], errors="raise")

    for column in CATEGORICAL_COLUMNS:
        data[column] = data[column].astype(str).str.strip()

    allowed_categories = {
        "Day": set(DAYS),
        "Meal": set(MEALS),
        "Menu": set(MENUS),
        "Holiday": set(YES_NO_VALUES),
        "Exam_Period": set(YES_NO_VALUES),
    }
    for column, allowed in allowed_categories.items():
        invalid = set(data[column].dropna()) - allowed
        if invalid:
            raise ValueError(f"{column} contains invalid values: {sorted(invalid)}")

    numeric_columns = [
        "Previous_Attendance",
        "Actual_Attendance",
        "Food_Prepared",
        "Food_Wasted",
    ]
    for column in numeric_columns:
        data[column] = pd.to_numeric(data[column], errors="raise")
        if data[column].isna().any():
            raise ValueError(f"{column} contains missing values")
        if (data[column] < 0).any():
            raise ValueError(f"{column} cannot contain negative values")

    if (data["Previous_Attendance"] > 5000).any():
        raise ValueError("Previous_Attendance cannot exceed 5000")
    if data[FEATURE_COLUMNS + [TARGET_COLUMN]].isna().any().any():
        raise ValueError("Training features and target must not contain missing values")

    return data


def prepare_features(data: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Return the model inputs and attendance target."""
    return data[FEATURE_COLUMNS].copy(), data[TARGET_COLUMN].copy()
