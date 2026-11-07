"""Model training and evaluation helpers for the unemployment dataset."""

from __future__ import annotations

from math import ceil
from typing import Any

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVR


MODEL_NAMES = (
    "Rừng ngẫu nhiên (Random forest)",
    "Tăng cường gradient (Gradient boosting)",
    "Hồi quy vector hỗ trợ (SVR - RBF)",
)


def _prepare_features(features: pd.DataFrame) -> pd.DataFrame:
    prepared = features.copy()
    for column in prepared.select_dtypes(include=["datetime", "datetimetz"]).columns:
        prepared[column] = prepared[column].astype("string")
    for column in prepared.columns:
        if not pd.api.types.is_numeric_dtype(prepared[column]):
            prepared[column] = prepared[column].astype(object)
            prepared[column] = prepared[column].where(
                prepared[column].notna(), np.nan
            )
    for column in prepared.select_dtypes(include="number").columns:
        prepared[column] = prepared[column].replace([np.inf, -np.inf], np.nan)
    prepared = prepared.dropna(axis=1, how="all")
    if prepared.empty or prepared.shape[1] == 0:
        raise ValueError("Không còn cột đặc trưng hợp lệ sau khi loại cột rỗng.")
    return prepared


def _make_pipeline(
    model_name: str,
    features: pd.DataFrame,
    random_state: int,
) -> Pipeline:
    categorical_columns = [
        column
        for column in features.columns
        if not pd.api.types.is_numeric_dtype(features[column])
        or pd.api.types.is_bool_dtype(features[column])
    ]
    numeric_columns = [
        column for column in features.columns if column not in categorical_columns
    ]

    scale_numeric = model_name == "Hồi quy vector hỗ trợ (SVR - RBF)"
    numeric_steps: list[tuple[str, Any]] = [
        ("imputer", SimpleImputer(strategy="median"))
    ]
    if scale_numeric:
        numeric_steps.append(("scaler", StandardScaler()))

    transformers: list[tuple[str, Pipeline, list[str]]] = []
    if numeric_columns:
        transformers.append(
            ("numeric", Pipeline(numeric_steps), numeric_columns)
        )
    if categorical_columns:
        transformers.append(
            (
                "categorical",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        (
                            "encoder",
                            OneHotEncoder(
                                handle_unknown="ignore", sparse_output=False
                            ),
                        ),
                    ]
                ),
                categorical_columns,
            )
        )

    preprocess = ColumnTransformer(transformers, remainder="drop")
    estimators = {
        "Rừng ngẫu nhiên (Random forest)": RandomForestRegressor(
            n_estimators=300,
            min_samples_leaf=2,
            random_state=random_state,
            n_jobs=-1,
        ),
        "Tăng cường gradient (Gradient boosting)": GradientBoostingRegressor(
            n_estimators=150,
            learning_rate=0.05,
            max_depth=2,
            loss="huber",
            random_state=random_state,
        ),
        "Hồi quy vector hỗ trợ (SVR - RBF)": SVR(
            kernel="rbf", C=10.0, epsilon=0.1
        ),
    }
    return Pipeline(
        [("preprocess", preprocess), ("model", estimators[model_name])]
    )


def _split_rows(
    frame: pd.DataFrame,
    test_size: float,
    split_strategy: str,
    random_state: int,
) -> tuple[np.ndarray, np.ndarray, str, str | None]:
    if split_strategy not in {"auto", "temporal", "random"}:
        raise ValueError("Chiến lược chia dữ liệu phải là auto, temporal hoặc random.")

    year_column = next(
        (column for column in frame.columns if column.lower() == "year"), None
    )
    years = frame[year_column] if year_column is not None else pd.Series(dtype=object)
    unique_years = years.dropna().unique() if year_column is not None else []
    use_temporal_split = split_strategy == "temporal" or (
        split_strategy == "auto" and len(unique_years) >= 3
    )

    if use_temporal_split:
        if year_column is None or len(unique_years) < 3:
            raise ValueError(
                "Để chia theo thời gian cần cột Year với ít nhất 3 năm khác nhau."
            )
        ordered_years = sorted(unique_years)
        test_year_count = max(1, ceil(len(ordered_years) * test_size))
        test_year_count = min(test_year_count, len(ordered_years) - 1)
        test_years = ordered_years[-test_year_count:]
        test_mask = frame[year_column].isin(test_years).to_numpy()
        test_indices = np.flatnonzero(test_mask)
        train_indices = np.flatnonzero(~test_mask)
        if not len(train_indices) or not len(test_indices):
            raise ValueError("Không thể tạo tập huấn luyện và tập kiểm tra theo năm.")
        return (
            train_indices,
            test_indices,
            "Theo thời gian",
            ", ".join(str(year) for year in test_years),
        )

    indices = np.arange(len(frame))
    train_indices, test_indices = train_test_split(
        indices, test_size=test_size, random_state=random_state
    )
    return train_indices, test_indices, "Ngẫu nhiên", None


def compare_regression_models(
    data: pd.DataFrame,
    target: str,
    feature_columns: list[str],
    *,
    test_size: float = 0.2,
    split_strategy: str = "auto",
    random_state: int = 42,
) -> dict[str, Any]:
    """Train and compare three regression pipelines on a held-out test set."""
    if target not in data.columns:
        raise ValueError(f"Không tìm thấy cột mục tiêu '{target}'.")
    if not 0.1 <= test_size <= 0.4:
        raise ValueError("Tỷ lệ tập kiểm tra phải nằm trong khoảng 0.1 đến 0.4.")

    selected_features = [column for column in feature_columns if column != target]
    missing_features = [column for column in selected_features if column not in data]
    if missing_features:
        raise ValueError(f"Không tìm thấy cột đặc trưng: {missing_features}.")
    if not selected_features:
        raise ValueError("Hãy chọn ít nhất một cột đặc trưng.")

    frame = data.loc[:, selected_features + [target]].copy()
    frame[target] = pd.to_numeric(frame[target], errors="coerce")
    target_values = frame[target].to_numpy(dtype=float, na_value=np.nan)
    valid_target = np.isfinite(target_values)
    if not valid_target.any():
        raise ValueError("Cột mục tiêu phải chứa giá trị số hợp lệ để đánh giá hồi quy.")
    source_positions = np.flatnonzero(valid_target)
    frame = frame.loc[valid_target].reset_index(drop=True)
    frame[target] = target_values[valid_target]
    if len(frame) < 5:
        raise ValueError("Cần ít nhất 5 dòng có giá trị mục tiêu hợp lệ.")

    y = frame[target].astype(float)
    features = _prepare_features(frame[selected_features])
    train_indices, test_indices, split_label, test_period = _split_rows(
        frame, test_size, split_strategy, random_state
    )
    X_train, X_test = features.iloc[train_indices], features.iloc[test_indices]
    y_train, y_test = y.iloc[train_indices], y.iloc[test_indices]

    metrics: list[dict[str, float | str]] = []
    fitted_models: dict[str, Pipeline] = {}
    predictions: dict[str, np.ndarray] = {}
    for model_name in MODEL_NAMES:
        pipeline = _make_pipeline(model_name, features, random_state)
        pipeline.fit(X_train, y_train)
        predicted = pipeline.predict(X_test)
        metrics.append(
            {
                "Mô hình": model_name,
                "MAE": float(mean_absolute_error(y_test, predicted)),
                "RMSE": float(np.sqrt(mean_squared_error(y_test, predicted))),
                "R2": (
                    float(r2_score(y_test, predicted))
                    if len(y_test) >= 2
                    else float("nan")
                ),
            }
        )
        fitted_models[model_name] = pipeline
        predictions[model_name] = predicted

    comparison = pd.DataFrame(metrics).sort_values("RMSE").reset_index(drop=True)
    best_model_name = str(comparison.iloc[0]["Mô hình"])
    best_predictions = predictions[best_model_name]

    prediction_frame = data.iloc[source_positions[test_indices]].reset_index(drop=True).copy()
    prediction_frame["Giá trị thực tế"] = y_test.reset_index(drop=True)
    prediction_frame["Dự đoán"] = best_predictions
    prediction_frame["Sai số"] = (
        prediction_frame["Giá trị thực tế"] - prediction_frame["Dự đoán"]
    )
    for model_name, model_predictions in predictions.items():
        if model_name != best_model_name:
            prediction_frame[f"Dự đoán - {model_name}"] = model_predictions

    importance = pd.DataFrame(columns=["Đặc trưng", "Mức ảnh hưởng"])
    if len(y_test) >= 2:
        result = permutation_importance(
            fitted_models[best_model_name],
            X_test,
            y_test,
            n_repeats=5,
            random_state=random_state,
            scoring="neg_mean_absolute_error",
        )
        importance = (
            pd.DataFrame(
                {
                    "Đặc trưng": X_test.columns,
                    "Mức ảnh hưởng": result.importances_mean,
                }
            )
            .sort_values("Mức ảnh hưởng", ascending=False)
            .reset_index(drop=True)
        )

    return {
        "comparison": comparison,
        "predictions": prediction_frame,
        "feature_importance": importance,
        "best_model": best_model_name,
        "models": fitted_models,
        "split_strategy": split_label,
        "test_period": test_period,
        "train_rows": len(train_indices),
        "test_rows": len(test_indices),
    }
