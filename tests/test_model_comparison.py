import unittest

import pandas as pd

from src.ml_preprocessing import preprocess_dataset
from src.model_comparison import MODEL_NAMES, compare_regression_models


class CompareRegressionModelsTests(unittest.TestCase):
    def setUp(self) -> None:
        self.data = pd.DataFrame(
            {
                "Year": [year for year in range(2015, 2025) for _ in range(2)],
                "Country": ["A", "B"] * 10,
                "GDP_Growth": [float(year % 5) for year in range(20)],
                "Unemployment_Rate": [
                    4.0 + year * 0.12 + country_offset
                    for year in range(10)
                    for country_offset in (0.0, 0.6)
                ],
            }
        )

    def test_compares_three_models_with_latest_years_held_out(self) -> None:
        result = compare_regression_models(
            self.data,
            "Unemployment_Rate",
            ["Year", "Country", "GDP_Growth"],
            test_size=0.2,
        )

        self.assertEqual(len(result["comparison"]), 3)
        self.assertEqual(set(result["comparison"]["Mô hình"]), set(MODEL_NAMES))
        self.assertEqual(result["split_strategy"], "Theo thời gian")
        self.assertEqual(result["test_period"], "2023, 2024")
        self.assertEqual(result["test_rows"], 4)
        self.assertEqual(result["best_model"], result["comparison"].iloc[0]["Mô hình"])
        self.assertEqual(len(result["predictions"]), 4)

    def test_requires_at_least_one_feature(self) -> None:
        with self.assertRaisesRegex(ValueError, "ít nhất một cột đặc trưng"):
            compare_regression_models(self.data, "Unemployment_Rate", [])

    def test_random_split_can_be_selected(self) -> None:
        result = compare_regression_models(
            self.data,
            "Unemployment_Rate",
            ["Country", "GDP_Growth"],
            split_strategy="random",
        )

        self.assertEqual(result["split_strategy"], "Ngẫu nhiên")
        self.assertIsNone(result["test_period"])

    def test_preprocesses_data_before_training(self) -> None:
        dirty_data = self.data.copy()
        dirty_data.columns = [f" {column} " for column in dirty_data.columns]
        dirty_data[" Country "] = " " + dirty_data[" Country "] + " "
        dirty_data.loc[0, " GDP_Growth "] = float("inf")
        dirty_data.loc[1, " Country "] = "   "
        dirty_data = pd.concat(
            [dirty_data, dirty_data.iloc[[2]], pd.DataFrame([{
                " Country ": None,
                " Year ": None,
                " GDP_Growth ": None,
                " Unemployment_Rate ": None,
            }])],
            ignore_index=True,
        )

        processed, report = preprocess_dataset(dirty_data)
        result = compare_regression_models(
            processed,
            "Unemployment_Rate",
            ["Year", "Country", "GDP_Growth"],
        )

        self.assertEqual(report["removed_duplicate_rows"], 1)
        self.assertEqual(report["removed_empty_rows"], 1)
        self.assertEqual(report["replaced_infinite_values"], 1)
        self.assertEqual(report["missing_values"], 2)
        self.assertEqual(len(result["comparison"]), 3)

    def test_preprocessing_rejects_duplicate_column_names(self) -> None:
        data = pd.DataFrame([[1, 2]], columns=[" Year", "Year "])

        with self.assertRaisesRegex(ValueError, "Tên cột bị trùng"):
            preprocess_dataset(data)

    def test_predictions_keep_original_rows_when_targets_are_missing(self) -> None:
        data = self.data.assign(Row_ID=range(len(self.data)))
        data.loc[0, "Unemployment_Rate"] = None

        result = compare_regression_models(
            data,
            "Unemployment_Rate",
            ["Year", "Country", "GDP_Growth"],
        )

        self.assertEqual(result["predictions"]["Row_ID"].tolist(), [16, 17, 18, 19])


if __name__ == "__main__":
    unittest.main()
