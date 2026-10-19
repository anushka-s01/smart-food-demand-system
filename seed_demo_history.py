"""Populate the local app with clearly labelled synthetic demonstration rows."""

from __future__ import annotations

import argparse
from contextlib import closing
from datetime import date, datetime, time, timedelta

from data_processing import DATA_PATH, load_and_validate_data
from database import (
    delete_predictions_by_source,
    get_connection,
    get_prediction_history,
    init_db,
    save_prediction,
)
from food_logic import calculate_food_requirement
from model import predict_demand


MEAL_TIMES = {
    "Breakfast": time(7, 30),
    "Lunch": time(12, 30),
    "Dinner": time(19, 30),
}
SOURCE = "Synthetic demo"

INSERT_SQL = """
    INSERT INTO predictions (
        prediction_date, day, meal, menu, holiday, exam_period,
        previous_attendance, predicted_attendance, recommended_food,
        expected_wastage, wastage_risk, created_at, record_source
    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
"""


def _build_demo_rows(count: int) -> list[tuple]:
    """Build deterministic demonstration rows without touching the database."""
    if count < 1:
        raise ValueError("count must be at least 1")

    data = load_and_validate_data(DATA_PATH)
    if count > len(data):
        raise ValueError(f"count cannot exceed the {len(data)} available scenarios")

    start_date = date.today() - timedelta(days=(count - 1) // 3)
    rows = []
    for index, row in data.head(count).reset_index(drop=True).iterrows():
        prediction_date = start_date + timedelta(days=index // 3)
        day = prediction_date.strftime("%A")
        meal = str(row["Meal"])
        menu = str(row["Menu"])
        previous = int(row["Previous_Attendance"])
        predicted = predict_demand(
            day=day,
            meal=meal,
            menu=menu,
            holiday=str(row["Holiday"]),
            exam_period=str(row["Exam_Period"]),
            previous_attendance=previous,
        )
        recommended, potential_surplus, risk = calculate_food_requirement(predicted)
        created_at = datetime.combine(
            prediction_date, MEAL_TIMES[meal]
        ).strftime("%Y-%m-%d %H:%M:%S")
        rows.append(
            (
                prediction_date.isoformat(),
                day,
                meal,
                menu,
                str(row["Holiday"]),
                str(row["Exam_Period"]),
                previous,
                predicted,
                recommended,
                potential_surplus,
                risk,
                created_at,
                SOURCE,
            )
        )
    return rows


def seed_demo_history(count: int = 120, replace: bool = False) -> tuple[int, int, int]:
    """Add up to ``count`` repeat-safe demo rows and return added/skipped counts."""
    demo_rows = _build_demo_rows(count)
    init_db()
    removed = delete_predictions_by_source(SOURCE) if replace else 0
    history = get_prediction_history()
    existing_keys = {
        (
            str(row.prediction_date),
            str(row.meal),
            str(row.menu),
            int(row.previous_attendance),
        )
        for row in history.itertuples()
        if row.record_source == SOURCE
    }

    added = 0
    skipped = 0
    for row in demo_rows:
        key = (row[0], row[2], row[3], row[6])
        if key in existing_keys:
            skipped += 1
            continue
        save_prediction(*row)
        existing_keys.add(key)
        added += 1

    return added, skipped, removed


def ensure_demo_history(count: int = 300) -> tuple[int, int, int]:
    """Transaction-safely keep exactly ``count`` labelled demonstration rows."""
    init_db()

    with closing(get_connection()) as connection:
        existing_count = connection.execute(
            "SELECT COUNT(*) FROM predictions WHERE record_source = ?", (SOURCE,)
        ).fetchone()[0]
    if existing_count == count:
        return 0, 0, 0

    demo_rows = _build_demo_rows(count)
    with closing(get_connection()) as connection:
        connection.execute("PRAGMA busy_timeout = 30000")
        connection.execute("BEGIN IMMEDIATE")
        current_count = connection.execute(
            "SELECT COUNT(*) FROM predictions WHERE record_source = ?", (SOURCE,)
        ).fetchone()[0]
        if current_count == count:
            connection.rollback()
            return 0, 0, 0

        removed = connection.execute(
            "DELETE FROM predictions WHERE record_source = ?", (SOURCE,)
        ).rowcount
        connection.executemany(INSERT_SQL, demo_rows)
        connection.commit()
    return count, 0, int(removed)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--count", type=int, default=120)
    parser.add_argument(
        "--replace",
        action="store_true",
        help="Replace existing Synthetic demo rows while preserving user predictions.",
    )
    arguments = parser.parse_args()
    inserted, already_present, removed = seed_demo_history(
        arguments.count, replace=arguments.replace
    )
    print(
        f"Removed {removed} old synthetic demo records; added {inserted}; "
        f"skipped {already_present} existing records."
    )
    print("These records are labelled 'Synthetic demo' inside the application.")
