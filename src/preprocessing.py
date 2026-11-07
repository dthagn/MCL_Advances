"""
Preprocessing module for unemployment dashboard.
Handles data filtering, period classification, and KPI calculations.
"""

import pandas as pd
from typing import List, Optional, Tuple


def classify_period(year: int) -> str:
    """
    Classify year into period: Pre-COVID, COVID, or Recovery.
    
    Args:
        year: Year to classify
        
    Returns:
        Period string
    """
    if year <= 2019:
        return "Pre-COVID (2010-2019)"
    elif year <= 2021:
        return "COVID (2020-2021)"
    else:
        return "Recovery (2022-2024)"


def add_period_column(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add period column to dataframe based on year.
    
    Args:
        df: DataFrame with Year column
        
    Returns:
        DataFrame with added Period column
    """
    df = df.copy()
    df['Period'] = df['Year'].apply(classify_period)
    return df


def filter_by_country(df: pd.DataFrame, countries: Optional[List[str]] = None) -> pd.DataFrame:
    """
    Filter dataframe by selected countries.
    
    Args:
        df: DataFrame to filter
        countries: List of country names to include. If None, return all.
        
    Returns:
        Filtered DataFrame
    """
    if countries is None or len(countries) == 0:
        return df
    return df[df['Country'].isin(countries)].copy()


def filter_by_year(df: pd.DataFrame, years: Optional[List[int]] = None) -> pd.DataFrame:
    """
    Filter dataframe by selected years.
    
    Args:
        df: DataFrame to filter
        years: List of years to include. If None, return all.
        
    Returns:
        Filtered DataFrame
    """
    if years is None or len(years) == 0:
        return df
    return df[df['Year'].isin(years)].copy()


def filter_by_period(df: pd.DataFrame, periods: Optional[List[str]] = None) -> pd.DataFrame:
    """
    Filter dataframe by selected periods.
    
    Args:
        df: DataFrame to filter (must have Period column)
        periods: List of periods to include. If None, return all.
        
    Returns:
        Filtered DataFrame
    """
    if 'Period' not in df.columns:
        df = add_period_column(df)
    
    if periods is None or len(periods) == 0:
        return df
    return df[df['Period'].isin(periods)].copy()


def calculate_kpi_metrics(df: pd.DataFrame, selected_country: Optional[str] = None, 
                           selected_year: Optional[int] = None) -> dict:
    """
    Calculate KPI metrics based on filtered data.
    
    Args:
        df: DataFrame with unemployment data
        selected_country: Optional selected country for specific KPI
        selected_year: Optional selected year for specific KPI
        
    Returns:
        Dictionary with KPI metrics
    """
    metrics = {}
    
    # Filter data if selections provided
    filtered_df = df.copy()
    if selected_country:
        filtered_df = filtered_df[filtered_df['Country'] == selected_country]
    if selected_year:
        filtered_df = filtered_df[filtered_df['Year'] == selected_year]
    
    # Calculate metrics
    if len(filtered_df) > 0:
        # Selected unemployment rate (if specific country and year selected)
        if selected_country and selected_year:
            selected_row = filtered_df[filtered_df['Year'] == selected_year]
            if len(selected_row) > 0:
                metrics['selected_unemployment'] = selected_row['Unemployment_Rate'].values[0]
            else:
                metrics['selected_unemployment'] = None
        else:
            metrics['selected_unemployment'] = None
        
        # Average unemployment rate
        metrics['avg_unemployment'] = df['Unemployment_Rate'].mean()
        
        # Maximum unemployment year
        max_unemployment_row = df.loc[df['Unemployment_Rate'].idxmax()]
        metrics['max_unemployment_year'] = max_unemployment_row['Year']
        metrics['max_unemployment_rate'] = max_unemployment_row['Unemployment_Rate']
        
        # Average GDP growth
        metrics['avg_gdp_growth'] = df['GDP_Growth'].mean()
        
        # Average youth unemployment
        metrics['avg_youth_unemployment'] = df['Youth_Unemployment_Rate'].mean()
        
        # Average inflation
        metrics['avg_inflation'] = df['Inflation'].mean()
    else:
        metrics = {
            'selected_unemployment': None,
            'avg_unemployment': None,
            'max_unemployment_year': None,
            'max_unemployment_rate': None,
            'avg_gdp_growth': None,
            'avg_youth_unemployment': None,
            'avg_inflation': None
        }
    
    return metrics


def get_country_colors() -> dict:
    """
    Get consistent color mapping for countries.
    
    Returns:
        Dictionary mapping country names to colors
    """
    return {
        'Japan': '#1f77b4',
        'Philippines': '#ff7f0e',
        'South Korea': '#2ca02c',
        'Thailand': '#d62728',
        'Vietnam': '#9467bd'
    }


def get_available_countries(df: pd.DataFrame) -> List[str]:
    """
    Get list of available countries in the dataset.
    
    Args:
        df: DataFrame with Country column
        
    Returns:
        List of country names
    """
    return sorted(df['Country'].unique().tolist())


def get_available_years(df: pd.DataFrame) -> List[int]:
    """
    Get list of available years in the dataset.
    
    Args:
        df: DataFrame with Year column
        
    Returns:
        List of years
    """
    return sorted(df['Year'].unique().tolist())


def get_available_periods(df: pd.DataFrame) -> List[str]:
    """
    Get list of available periods in the dataset.
    
    Args:
        df: DataFrame with Year column
        
    Returns:
        List of period names
    """
    years = get_available_years(df)
    periods = set()
    for year in years:
        periods.add(classify_period(year))
    return sorted(list(periods))
