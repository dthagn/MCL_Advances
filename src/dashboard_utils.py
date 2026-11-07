"""
Dashboard utility functions for unemployment dashboard.
Helper functions for displaying data, formatting, and error handling.
"""

import streamlit as st
import pandas as pd
from typing import Optional


def display_kpi_card(title: str, value: Optional[float], unit: str = "%", 
                     delta: Optional[float] = None, help_text: Optional[str] = None):
    """
    Display a KPI card with metric value.
    
    Args:
        title: Title of the KPI
        value: Value to display
        unit: Unit of measurement
        delta: Optional delta value to show change
        help_text: Optional help text tooltip
    """
    if value is None:
        st.metric(title, "N/A", help=help_text)
    else:
        if delta is not None:
            st.metric(title, f"{value:.2f} {unit}", f"{delta:+.2f} {unit}", help=help_text)
        else:
            st.metric(title, f"{value:.2f} {unit}", help=help_text)


def display_data_table(df: pd.DataFrame, title: str = "Dữ liệu", height: int = 300):
    """
    Display a data table with title.
    
    Args:
        df: DataFrame to display
        title: Title for the table
        height: Height of the table
    """
    st.subheader(title)
    st.dataframe(df, use_container_width=True, height=height)


def display_correlation_table(correlation_df: pd.DataFrame):
    """
    Display correlation results in a formatted table.
    
    Args:
        correlation_df: DataFrame with correlation results
    """
    if len(correlation_df) == 0:
        st.warning("Không có dữ liệu tương quan.")
        return
    
    # Format correlation values
    display_df = correlation_df.copy()
    display_df['Correlation'] = display_df['Correlation'].apply(lambda x: f"{x:.4f}")
    
    st.subheader("Kết quả phân tích tương quan")
    st.dataframe(display_df, use_container_width=True, hide_index=True)


def display_findings_section(findings_content: str):
    """
    Display findings from analysis in a formatted section.
    
    Args:
        findings_content: Markdown content of findings
    """
    st.markdown("---")
    st.header("Kết quả phân tích")
    st.markdown(findings_content)


def display_data_sources(source_metadata_df: pd.DataFrame, data_dictionary_df: pd.DataFrame):
    """
    Display data sources and definitions.
    
    Args:
        source_metadata_df: DataFrame with source metadata
        data_dictionary_df: DataFrame with data dictionary
    """
    st.markdown("---")
    st.header("Nguồn dữ liệu")
    
    # Display source metadata
    st.subheader("Thông tin nguồn dữ liệu")
    
    # Filter out unavailable variables
    available_sources = source_metadata_df[~source_metadata_df['notes'].str.contains('NOT AVAILABLE', na=False)]
    
    for _, row in available_sources.iterrows():
        with st.expander(f"{row['variable']} - {row['source']}"):
            st.write(f"**URL:** {row['URL']}")
            st.write(f"**Indicator Code:** {row['indicator_code']}")
            st.write(f"**Definition:** {row['definition']}")
            st.write(f"**Notes:** {row['notes']}")
    
    # Display data dictionary
    st.subheader("Từ điển dữ liệu")
    st.dataframe(data_dictionary_df, use_container_width=True, hide_index=True)


def display_limitations():
    """
    Display data limitations section.
    """
    st.markdown("---")
    st.header("Hạn chế dữ liệu")
    
    limitations = [
        "**Dữ liệu giới tính:** Tỷ lệ thất nghiệp theo giới tính (nam/nữ) không có sẵn do yêu cầu xác thực API của ILOSTAT và hạn chế của API World Bank cho các quốc gia này.",
        "**Dữ liệu trình độ học vấn:** Các biến liên quan đến trình độ học vấn không được đưa vào bộ dữ liệu cuối cùng do không có sẵn đáng tin cậy cho tất cả các quốc gia.",
        "**Phạm vi thời gian:** Phân tích giới hạn trong giai đoạn 2010-2024. Không thể đánh giá xu hướng dài hạn.",
        "**Tương quan vs Nhân quả:** Tất cả các phân tích tương quan chỉ cho thấy mối quan hệ thống kê. Không thể đưa ra kết luận về quan hệ nhân quả dựa trên dữ liệu này.",
        "**Kích thước mẫu:** Các tương quan ở cấp quốc gia dựa trên chỉ 15 điểm dữ liệu cho mỗi quốc gia, làm hạn chế sức mạnh thống kê.",
        "**Các yếu tố bên ngoài:** Phân tích không tính đến các other yếu tố kinh tế, xã hội hoặc chính sách có thể ảnh hưởng đến thất nghiệp.",
        "**Nguồn dữ liệu:** Tất cả dữ liệu đến từ ước tính của World Bank. Các ước tính mô hình hóa của ILO được sử dụng cho thất nghiệp và tỷ lệ tham gia lực lượng lao động, có thể có những hạn chế vốn có.",
        "**Phạm vi địa lý:** Chỉ có 5 quốc gia châu Á được đưa vào. Kết quả có thể không tổng quát hóa cho các khu vực khác.",
        "**Định nghĩa biến:** Tất cả các biến sử dụng định nghĩa của World Bank. Các định nghĩa khác nhau có thể cho ra kết quả khác nhau."
    ]
    
    for i, limitation in enumerate(limitations, 1):
        st.markdown(f"{i}. {limitation}")


def display_methodology():
    """
    Display methodology section.
    """
    st.markdown("---")
    st.header("Phương pháp")
    
    methodology = """
    ### Phạm vi nghiên cứu
    - **Quốc gia:** 5 quốc gia châu Á (Nhật Bản, Philippines, Hàn Quốc, Thái Lan, Việt Nam)
    - **Thời gian:** 2010-2024
    - **Số lượng quan sát:** 75 (5 quốc gia × 15 năm)
    
    ### Các chỉ số phân tích
    - Trung bình (Mean)
    - Trung vị (Median)
    - Giá trị nhỏ nhất (Minimum)
    - Giá trị lớn nhất (Maximum)
    - Độ lệch chuẩn (Standard Deviation)
    - Thay đổi theo thời gian
    - Chênh lệch giới tính (Gender Gap) - *không có sẵn*
    - Chênh lệch thanh niên (Youth Gap)
    - Hệ số tương quan Pearson (Pearson Correlation)
    
    ### Công thức tính toán
    
    **Youth Gap:**
    ```
    Youth Gap = Youth_Unemployment_Rate - Unemployment_Rate
    ```
    
    **Hệ số tương quan Pearson:**
    ```
    r = Σ((Xi - X̄)(Yi - Ỹ)) / √(Σ(Xi - X̄)² × Σ(Yi - Ỹ)²)
    ```
    
    ### Phân loại giai đoạn
    - **Trước COVID-19 (2010-2019):** Giai đoạn bình thường
    - **COVID-19 (2020-2021):** Giai đoạn đại dịch
    **Phục hồi (2022-2024):** Giai đoạn phục hồi
    
    ### Lưu ý quan trọng
    - Tương quan không đồng nghĩa với nhân quả
    - Kết quả chỉ phản ánh mối quan hệ thống kê
    - Không thể suy rộng kết quả cho các khu vực khác
    - Dữ liệu dựa trên định nghĩa của World Bank
    """
    
    st.markdown(methodology)


def display_error_message(message: str):
    """
    Display an error message to the user.
    
    Args:
        message: Error message to display
    """
    st.error(message)


def display_warning_message(message: str):
    """
    Display a warning message to the user.
    
    Args:
        message: Warning message to display
    """
    st.warning(message)


def display_info_message(message: str):
    """
    Display an info message to the user.
    
    Args:
        message: Info message to display
    """
    st.info(message)


def format_number(value: float, decimals: int = 2) -> str:
    """
    Format a number with specified decimal places.
    
    Args:
        value: Number to format
        decimals: Number of decimal places
        
    Returns:
        Formatted string
    """
    return f"{value:.{decimals}f}"


def get_period_from_year(year: int) -> str:
    """
    Get period name from year.
    
    Args:
        year: Year to classify
        
    Returns:
        Period name
    """
    if year <= 2019:
        return "Trước COVID-19 (2010-2019)"
    elif year <= 2021:
        return "COVID-19 (2020-2021)"
    else:
        return "Phục hồi (2022-2024)"
