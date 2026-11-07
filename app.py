from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st

from src.ml_preprocessing import preprocess_dataset
from src.model_comparison import MODEL_NAMES, compare_regression_models


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_DATA_PATHS = (
    PROJECT_ROOT / "data" / "processed" / "unemployment_final.csv",
    PROJECT_ROOT / "src" / "data" / "processed" / "unemployment_final.csv",
)


@st.cache_data
def load_data_path(path: str) -> pd.DataFrame:
    return pd.read_csv(path)


def available_data_paths() -> list[str]:
    candidates = [path for path in DEFAULT_DATA_PATHS if path.is_file()]
    source_folder = PROJECT_ROOT / "src"
    if source_folder.is_dir():
        candidates.extend(
            path
            for path in sorted(source_folder.rglob("*.csv"))
            if path not in candidates
        )
    return [str(path) for path in candidates]


@st.cache_data
def load_uploaded_data(file_bytes: bytes) -> pd.DataFrame:
    from io import BytesIO

    return pd.read_csv(BytesIO(file_bytes))


st.set_page_config(
    page_title="So sánh mô hình học máy",
    page_icon=":material/insights:",
    layout="wide",
)
st.title("Đánh giá và so sánh mô hình học máy")
st.markdown(
    "Huấn luyện ba mô hình hồi quy trên dữ liệu kinh tế, đánh giá trên tập kiểm tra "
    "chưa được dùng khi huấn luyện và xếp hạng theo RMSE."
)

data_paths = available_data_paths()
uploaded_file = st.sidebar.file_uploader(
    "Tải lên dữ liệu CSV", type=["csv"], help="CSV cần có cột mục tiêu dạng số."
)

if uploaded_file is not None:
    try:
        data = load_uploaded_data(uploaded_file.getvalue())
        data_source = uploaded_file.name
    except (
        UnicodeDecodeError,
        pd.errors.EmptyDataError,
        pd.errors.ParserError,
        ValueError,
    ) as error:
        st.error(f"Không thể đọc CSV: {error}")
        st.stop()
elif data_paths:
    if len(data_paths) == 1:
        selected_path = data_paths[0]
    else:
        selected_path = st.sidebar.selectbox(
            "Chọn CSV có sẵn",
            data_paths,
            format_func=lambda path: str(Path(path).relative_to(PROJECT_ROOT)),
        )
    try:
        data = load_data_path(str(selected_path))
        data_source = str(Path(selected_path).relative_to(PROJECT_ROOT))
    except (
        OSError,
        UnicodeDecodeError,
        pd.errors.EmptyDataError,
        pd.errors.ParserError,
        ValueError,
    ) as error:
        st.error(f"Không thể đọc CSV `{selected_path}`: {error}")
        st.stop()
else:
    st.info(
        "Chưa tìm thấy CSV trong `src/` hoặc tại đường dẫn dữ liệu mặc định. "
        "Tải lên CSV ở thanh bên để bắt đầu."
    )
    st.stop()

try:
    data, cleaning_report = preprocess_dataset(data)
except ValueError as error:
    st.error(f"Không thể tiền xử lý dữ liệu: {error}")
    st.stop()

st.caption(
    f"Nguồn dữ liệu: `{data_source}` · "
    f"{cleaning_report['original_rows']:,} dòng ban đầu · "
    f"{len(data):,} dòng sau tiền xử lý · {len(data.columns)} cột"
)
with st.expander("Tiền xử lý dữ liệu", expanded=True):
    st.markdown(
        "Đã chuẩn hóa khoảng trắng trong tên cột và dữ liệu phân loại, chuyển giá trị "
        "vô cực thành thiếu, rồi loại dòng rỗng và dòng trùng lặp. "
        "Giá trị thiếu còn lại được điền trong pipeline: trung vị cho biến số, "
        "giá trị phổ biến nhất cho biến phân loại. Mã hóa one-hot áp dụng cho biến "
        "phân loại; SVR chuẩn hóa biến số. Các tham số điền và chuẩn hóa chỉ được "
        "học từ tập huấn luyện để tránh rò rỉ dữ liệu."
    )
    report_columns = st.columns(4)
    report_columns[0].metric(
        "Dòng rỗng đã loại", cleaning_report["removed_empty_rows"]
    )
    report_columns[1].metric(
        "Dòng trùng đã loại", cleaning_report["removed_duplicate_rows"]
    )
    report_columns[2].metric(
        "Khoảng trắng đã chuẩn hóa", cleaning_report["trimmed_text_values"]
    )
    report_columns[3].metric(
        "Giá trị thiếu còn lại", cleaning_report["missing_values"]
    )
    if cleaning_report["removed_empty_columns"]:
        st.caption(
            "Cột hoàn toàn rỗng đã loại: "
            + ", ".join(cleaning_report["removed_empty_columns"])
        )
    if cleaning_report["replaced_infinite_values"]:
        st.caption(
            f"Đã chuyển {cleaning_report['replaced_infinite_values']} giá trị vô cực "
            "thành giá trị thiếu để pipeline xử lý."
        )

    st.subheader("Bản xem trước dữ liệu sau tiền xử lý")
    st.dataframe(
        data.head(20),
        width="stretch",
        hide_index=True,
        alt="Bản xem trước dữ liệu đã làm sạch trước khi huấn luyện",
    )
    st.download_button(
        "Tải dữ liệu đã làm sạch",
        data=data.to_csv(index=False).encode("utf-8-sig"),
        file_name="du_lieu_da_tien_xu_ly.csv",
        mime="text/csv",
        icon=":material/download:",
    )

numeric_columns = data.select_dtypes(include="number").columns.tolist()
if not numeric_columns:
    st.error("Không có cột số để làm biến mục tiêu cho bài toán hồi quy.")
    st.stop()

preferred_target = next(
    (column for column in ("Unemployment_Rate", "unemployment_rate") if column in numeric_columns),
    numeric_columns[0],
)
with st.sidebar:
    st.header("Thiết lập đánh giá")
    target = st.selectbox(
        "Biến mục tiêu",
        numeric_columns,
        index=numeric_columns.index(preferred_target),
    )
    available_features = [column for column in data.columns if column != target]
    default_features = available_features.copy()
    if "Country" in default_features and "Country_Code" in default_features:
        default_features.remove("Country_Code")
    feature_columns = st.multiselect(
        "Đặc trưng đầu vào",
        available_features,
        default=default_features,
        help=(
            "Chọn các thông tin sẵn có tại thời điểm dự đoán. Country_Code được "
            "bỏ khỏi mặc định vì trùng lặp thông tin với Country."
        ),
    )
    split_strategy = st.selectbox(
        "Cách chia tập dữ liệu",
        ("Tự động (ưu tiên theo năm)", "Theo thời gian", "Ngẫu nhiên"),
    )
    test_size = st.slider("Tỷ lệ tập kiểm tra", 0.1, 0.4, 0.2, 0.05)

strategy_values = {
    "Tự động (ưu tiên theo năm)": "auto",
    "Theo thời gian": "temporal",
    "Ngẫu nhiên": "random",
}

if not feature_columns:
    st.warning("Hãy chọn ít nhất một đặc trưng đầu vào.")
    st.stop()

with st.spinner("Đang huấn luyện và đánh giá ba mô hình..."):
    try:
        result = compare_regression_models(
            data,
            target,
            feature_columns,
            test_size=test_size,
            split_strategy=strategy_values[split_strategy],
        )
    except (ValueError, TypeError) as error:
        st.error(f"Không thể đánh giá mô hình: {error}")
        st.stop()

best_row = result["comparison"].iloc[0]
metric_columns = st.columns(4)
metric_columns[0].metric("Mô hình tốt nhất", result["best_model"])
metric_columns[1].metric("RMSE", f"{best_row['RMSE']:.4f}")
metric_columns[2].metric("MAE", f"{best_row['MAE']:.4f}")
metric_columns[3].metric(
    "R²",
    f"{best_row['R2']:.4f}" if pd.notna(best_row["R2"]) else "Không đủ mẫu",
)

test_description = (
    f" · Năm kiểm tra: {result['test_period']}" if result["test_period"] else ""
)
st.caption(
    f"Chia {result['split_strategy']} · {result['train_rows']} dòng huấn luyện · "
    f"{result['test_rows']} dòng kiểm tra{test_description}. "
    "RMSE và MAE càng thấp càng tốt; R² càng cao càng tốt."
)

st.subheader("Bảng xếp hạng mô hình")
st.dataframe(
    result["comparison"].style.format(
        {"MAE": "{:.4f}", "RMSE": "{:.4f}", "R2": "{:.4f}"}
    ),
    width="stretch",
    hide_index=True,
)

left, right = st.columns(2)
with left:
    st.subheader("RMSE theo mô hình")
    st.bar_chart(
        result["comparison"].set_index("Mô hình")[["RMSE"]],
        y_label="RMSE (thấp hơn tốt hơn)",
        alt="So sánh sai số RMSE của ba mô hình, giá trị thấp hơn là tốt hơn",
    )
with right:
    st.subheader("Thực tế và dự đoán")
    chart_data = result["predictions"][["Giá trị thực tế", "Dự đoán"]]
    st.scatter_chart(
        chart_data,
        x="Giá trị thực tế",
        y="Dự đoán",
        height=320,
        alt="So sánh giá trị thực tế với dự đoán của mô hình tốt nhất",
    )

if not result["feature_importance"].empty:
    st.subheader("Ảnh hưởng của đặc trưng lên mô hình tốt nhất")
    st.caption(
        "Độ quan trọng được ước lượng bằng permutation importance trên tập kiểm tra; "
        "đây không phải bằng chứng về quan hệ nhân quả."
    )
    st.bar_chart(
        result["feature_importance"].set_index("Đặc trưng"),
        y="Mức ảnh hưởng",
        alt="Mức ảnh hưởng của từng đặc trưng lên sai số mô hình",
    )

st.subheader("Dự đoán trên tập kiểm tra")
st.dataframe(
    result["predictions"],
    width="stretch",
    hide_index=True,
    alt="Giá trị thực tế, dự đoán và sai số của các dòng kiểm tra",
)

st.download_button(
    "Tải bảng dự đoán CSV",
    data=result["predictions"].to_csv(index=False).encode("utf-8-sig"),
    file_name="du_doan_tap_kiem_tra.csv",
    mime="text/csv",
    icon=":material/download:",
)

st.subheader("Ba mô hình được đánh giá")
st.write(", ".join(MODEL_NAMES))
st.caption(
    "Đây là so sánh trên một tập kiểm tra cố định, không thay thế kiểm định ngoài "
    "mẫu hoặc xác nhận chuyên môn. Mô hình tốt nhất được chọn theo RMSE."
)
