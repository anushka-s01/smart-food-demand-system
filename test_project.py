"""Practical tests for the prediction, food, dataset, and database layers."""

from __future__ import annotations

import os
from pathlib import Path
import tempfile
import unittest

from data_processing import DATA_PATH, load_and_validate_data, prepare_features
from database import get_prediction_history, init_db, save_prediction
from food_logic import calculate_food_requirement
from model import predict_demand, prediction_engine_status
from seed_demo_history import seed_demo_history_if_empty


class DatasetTests(unittest.TestCase):
    def test_dataset_contract(self) -> None:
        data = load_and_validate_data(DATA_PATH)
        features, target = prepare_features(data)
        self.assertEqual(len(data), 300)
        self.assertEqual(len(features), len(target))
        self.assertFalse(features.isna().any().any())
        self.assertFalse(target.isna().any())


class PredictionTests(unittest.TestCase):
    def test_valid_prediction(self) -> None:
        predicted = predict_demand(
            "Monday", "Lunch", "Biryani", "No", "No", 500
        )
        self.assertIsInstance(predicted, int)
        self.assertGreaterEqual(predicted, 0)

    def test_zero_previous_attendance(self) -> None:
        predicted = predict_demand(
            "Monday", "Breakfast", "Poha", "No", "No", 0
        )
        self.assertGreaterEqual(predicted, 0)

    def test_holiday_and_exam_cases(self) -> None:
        for holiday in ("No", "Yes"):
            for exam_period in ("No", "Yes"):
                result = predict_demand(
                    "Friday", "Dinner", "Roti Sabzi", holiday, exam_period, 450
                )
                self.assertIsInstance(result, int)

    def test_invalid_inputs_are_rejected(self) -> None:
        with self.assertRaises(ValueError):
            predict_demand("Monday", "Lunch", "Biryani", "No", "No", -1)
        with self.assertRaises(ValueError):
            predict_demand("Monday", "Lunch", "Unknown", "No", "No", 500)

    def test_engine_status_is_explicit(self) -> None:
        name, detail = prediction_engine_status()
        self.assertIn(name, {"Random Forest model", "Historical-average fallback"})
        self.assertTrue(detail)


class FoodLogicTests(unittest.TestCase):
    def test_known_example(self) -> None:
        self.assertEqual(calculate_food_requirement(500), (525, 25, "Low"))
        self.assertEqual(calculate_food_requirement(457), (480, 23, "Low"))

    def test_zero_and_negative_attendance(self) -> None:
        self.assertEqual(calculate_food_requirement(0), (0, 0, "Low"))
        with self.assertRaises(ValueError):
            calculate_food_requirement(-1)


class DatabaseTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        os.environ["FOOD_DEMAND_DB_PATH"] = str(
            Path(self.temporary_directory.name) / "test.db"
        )

    def tearDown(self) -> None:
        os.environ.pop("FOOD_DEMAND_DB_PATH", None)
        self.temporary_directory.cleanup()

    def test_empty_history_and_round_trip(self) -> None:
        init_db()
        self.assertTrue(get_prediction_history().empty)
        row_id = save_prediction(
            "2026-10-19",
            "Monday",
            "Lunch",
            "Biryani",
            "No",
            "No",
            500,
            520,
            546,
            26,
            "Low",
            "2026-10-19 12:00:00",
        )
        self.assertEqual(row_id, 1)
        history = get_prediction_history()
        self.assertEqual(len(history), 1)
        self.assertEqual(int(history.iloc[0]["predicted_attendance"]), 520)

    def test_fresh_database_is_seeded_once(self) -> None:
        added, skipped, removed = seed_demo_history_if_empty(count=300)
        self.assertEqual((added, skipped, removed), (300, 0, 0))
        history = get_prediction_history()
        self.assertEqual(len(history), 300)
        self.assertEqual(set(history["record_source"]), {"Synthetic demo"})

        second_run = seed_demo_history_if_empty(count=300)
        self.assertEqual(second_run, (0, 0, 0))
        self.assertEqual(len(get_prediction_history()), 300)


if __name__ == "__main__":
    unittest.main(verbosity=2)
