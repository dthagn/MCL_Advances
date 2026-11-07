"""Data cleaning helpers shared by the machine-learning workflow."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def preprocess_dataset(data: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Clean tabular data and report changes before model-specific preprocessing."""
    if data.empty:
        raise ValueError("Tệp dữ liệu không có dòng nào.")

    processed = data.copy()
    original_rows, original_columns = processed.shape
    processed.columns = [
        column.strip() if isinstance(column, str) else column
        for column in processed.columns
    ]
    if processed.columns.duplicated().any():
        duplicates = processed.columns[processed.columns.duplicated()].tolist()
        raise ValueError(f"Tên cột bị trùng sau khi loại khoảng trắng: {duplicates}.")

    trimmed_values = 0
    for column in processed.columns:
        series = processed[column]
        if pd.api.types.is_object_dtype(series.dtype):
            non_missing = series.dropna()
            is_text = not non_missing.empty and non_missing.map(
                lambda value: isinstance(value, str)
            ).all()
        else:
            is_text = (
                pd.api.types.is_string_dtype(series.dtype)
                or isinstance(series.dtype, pd.CategoricalDtype)
            )
        if not is_text:
            continue

        original = processed[column].astype("string")
        trimmed = original.str.strip()
        trimmed_values += int((original != trimmed).fillna(False).sum())
        processed[column] = trimmed.replace("", pd.NA)

    for column in processed.columns:
        if not pd.api.types.is_object_dtype(processed[column].dtype):
            continue
        non_missing = processed[column].notna()
        converted = pd.to_numeric(processed[column], errors="coerce")
        if non_missing.any() and converted.loc[non_missing].notna().all():
            processed[column] = converted

    infinite_values = 0
    for column in processed.select_dtypes(include="number").columns:
        values = processed[column].to_numpy(dtype=float, na_value=np.nan)
        infinite_values += int(np.isinf(values).sum())
        processed[column] = processed[column].replace([np.inf, -np.inf], np.nan)

    empty_columns = processed.columns[processed.isna().all()].tolist()
    if empty_columns:
        processed = processed.drop(columns=empty_columns)

    empty_rows = int(processed.isna().all(axis=1).sum())
    processed = processed.dropna(how="all")
    duplicate_rows = int(processed.duplicated().sum())
    processed = processed.drop_duplicates().reset_index(drop=True)
    if processed.empty or processed.shape[1] == 0:
        raise ValueError("Dữ liệu không còn dòng hoặc cột hợp lệ sau tiền xử lý.")

    report: dict[str, Any] = {
        "original_rows": original_rows,
        "processed_rows": len(processed),
        "removed_empty_rows": empty_rows,
        "removed_duplicate_rows": duplicate_rows,
        "trimmed_text_values": trimmed_values,
        "replaced_infinite_values": infinite_values,
        "removed_empty_columns": empty_columns,
        "missing_values": int(processed.isna().sum().sum()),
        "original_columns": original_columns,
        "processed_columns": len(processed.columns),
    }
    return processed, report
