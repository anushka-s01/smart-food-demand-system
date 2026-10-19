"""SQLite storage for prediction history."""

from __future__ import annotations

import os
from pathlib import Path
import sqlite3
from contextlib import closing

import pandas as pd


DEFAULT_DB_PATH = Path(__file__).with_name("food_demand.db")


def _database_path() -> Path:
    configured = os.environ.get("FOOD_DEMAND_DB_PATH")
    return Path(configured) if configured else DEFAULT_DB_PATH


def get_connection() -> sqlite3.Connection:
    path = _database_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    return sqlite3.connect(path, timeout=30)


def init_db() -> None:
    with closing(get_connection()) as connection:
        with connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS predictions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    prediction_date TEXT NOT NULL,
                    day TEXT NOT NULL,
                    meal TEXT NOT NULL,
                    menu TEXT NOT NULL,
                    holiday TEXT NOT NULL,
                    exam_period TEXT NOT NULL,
                    previous_attendance INTEGER NOT NULL CHECK(previous_attendance >= 0),
                    predicted_attendance INTEGER NOT NULL CHECK(predicted_attendance >= 0),
                    recommended_food INTEGER NOT NULL CHECK(recommended_food >= 0),
                    expected_wastage INTEGER NOT NULL CHECK(expected_wastage >= 0),
                    wastage_risk TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    record_source TEXT NOT NULL DEFAULT 'User prediction'
                )
                """
            )
            existing_columns = {
                row[1]
                for row in connection.execute("PRAGMA table_info(predictions)").fetchall()
            }
            if "record_source" not in existing_columns:
                connection.execute(
                    "ALTER TABLE predictions ADD COLUMN record_source TEXT "
                    "NOT NULL DEFAULT 'User prediction'"
                )


def save_prediction(
    prediction_date,
    day,
    meal,
    menu,
    holiday,
    exam_period,
    previous_attendance,
    predicted_attendance,
    recommended_food,
    expected_wastage,
    wastage_risk,
    created_at,
    record_source="User prediction",
) -> int:
    values = (
        str(prediction_date),
        str(day),
        str(meal),
        str(menu),
        str(holiday),
        str(exam_period),
        int(previous_attendance),
        int(predicted_attendance),
        int(recommended_food),
        int(expected_wastage),
        str(wastage_risk),
        str(created_at),
        str(record_source),
    )
    if any(value < 0 for value in values[6:10]):
        raise ValueError("Attendance and portion values cannot be negative")

    with closing(get_connection()) as connection:
        with connection:
            cursor = connection.execute(
                """
                INSERT INTO predictions (
                    prediction_date, day, meal, menu, holiday, exam_period,
                    previous_attendance, predicted_attendance, recommended_food,
                    expected_wastage, wastage_risk, created_at, record_source
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                values,
            )
            return int(cursor.lastrowid)


def get_prediction_history() -> pd.DataFrame:
    columns = [
        "id",
        "prediction_date",
        "day",
        "meal",
        "menu",
        "holiday",
        "exam_period",
        "previous_attendance",
        "predicted_attendance",
        "recommended_food",
        "expected_wastage",
        "wastage_risk",
        "created_at",
        "record_source",
    ]
    with closing(get_connection()) as connection:
        rows = connection.execute(
            """
            SELECT id, prediction_date, day, meal, menu, holiday, exam_period,
                   previous_attendance, predicted_attendance, recommended_food,
                   expected_wastage, wastage_risk, created_at, record_source
            FROM predictions
            ORDER BY id DESC
            """
        ).fetchall()
    return pd.DataFrame(rows, columns=columns)


def delete_predictions_by_source(record_source: str) -> int:
    """Delete only records carrying the specified source label."""
    with closing(get_connection()) as connection:
        with connection:
            cursor = connection.execute(
                "DELETE FROM predictions WHERE record_source = ?",
                (str(record_source),),
            )
            return int(cursor.rowcount)


if __name__ == "__main__":
    init_db()
    print(f"Database initialized at {_database_path()}")
