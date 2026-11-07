"""
Visualization module for unemployment dashboard.
Creates Plotly charts for all analysis visualizations.
"""

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from typing import Optional, List


def create_unemployment_trend(df: pd.DataFrame, countries: Optional[List[str]] = None) -> go.Figure:
    """
    Create unemployment trend line chart.
    
    Args:
        df: DataFrame with unemployment data
        countries: List of countries to include. If None, include all.
        
    Returns:
        Plotly figure
    """
    if countries:
        df = df[df['Country'].isin(countries)]
    
    if len(df) == 0:
        fig = go.Figure()
        fig.add_annotation(text="Không đủ dữ liệu để hiển thị biểu đồ này.",
                          xref="paper", yref="paper",
                          x=0.5, y=0.5, showarrow=False)
        return fig
    
    country_colors = {
        'Japan': '#1f77b4',
        'Philippines': '#ff7f0e',
        'South Korea': '#2ca02c',
        'Thailand': '#d62728',
        'Vietnam': '#9467bd'
    }
    
    fig = go.Figure()
    
    for country in sorted(df['Country'].unique()):
        country_data = df[df['Country'] == country].sort_values('Year')
        fig.add_trace(go.Scatter(
            x=country_data['Year'],
            y=country_data['Unemployment_Rate'],
            mode='lines+markers',
            name=country,
            line=dict(color=country_colors.get(country, '#333333')),
            hovertemplate='<b>%{fullData.name}</b><br>Year: %{x}<br>Unemployment: %{y:.2f}%<extra></extra>'
        ))
    
    fig.update_layout(
        title='Xu hướng tỷ lệ thất nghiệp theo thời gian (2010-2024)',
        xaxis_title='Năm',
        yaxis_title='Tỷ lệ thất nghiệp (%)',
        hovermode='x unified',
        template='plotly_white',
        height=400
    )
    
    return fig


def create_country_comparison(df: pd.DataFrame, metric: str = 'Unemployment_Rate') -> go.Figure:
    """
    Create country comparison bar chart.
    
    Args:
        df: DataFrame with unemployment data
        metric: Metric to compare (default: Unemployment_Rate)
        
    Returns:
        Plotly figure
    """
    if len(df) == 0:
        fig = go.Figure()
        fig.add_annotation(text="Không đủ dữ liệu để hiển thị biểu đồ này.",
                          xref="paper", yref="paper",
                          x=0.5, y=0.5, showarrow=False)
        return fig
    
    # Calculate average by country
    country_avg = df.groupby('Country')[metric].mean().reset_index()
    country_avg = country_avg.sort_values(metric, ascending=True)
    
    country_colors = {
        'Japan': '#1f77b4',
        'Philippines': '#ff7f0e',
        'South Korea': '#2ca02c',
        'Thailand': '#d62728',
        'Vietnam': '#9467bd'
    }
    
    colors = [country_colors.get(country, '#333333') for country in country_avg['Country']]
    
    fig = go.Figure(data=go.Bar(
        x=country_avg[metric],
        y=country_avg['Country'],
        orientation='h',
        marker_color=colors,
        text=country_avg[metric].round(2),
        textposition='outside'
    ))
    
    metric_label = 'Tỷ lệ thất nghiệp (%)' if metric == 'Unemployment_Rate' else metric
    
    fig.update_layout(
        title=f'So sánh trung bình {metric_label} theo quốc gia (2010-2024)',
        xaxis_title=metric_label,
        yaxis_title='Quốc gia',
        template='plotly_white',
        height=400
    )
    
    return fig


def create_choropleth_map(df: pd.DataFrame, year: Optional[int] = None) -> go.Figure:
    """
    Create choropleth map of unemployment rates.
    
    Args:
        df: DataFrame with unemployment data
        year: Specific year to display. If None, use average.
        
    Returns:
        Plotly figure
    """
    if len(df) == 0:
        fig = go.Figure()
        fig.add_annotation(text="Không đủ dữ liệu để hiển thị biểu đồ này.",
                          xref="paper", yref="paper",
                          x=0.5, y=0.5, showarrow=False)
        return fig
    
    # Filter by year if specified
    if year:
        map_data = df[df['Year'] == year].copy()
        title_suffix = f'Năm {year}'
    else:
        # Use average across all years
        map_data = df.groupby('Country')['Unemployment_Rate'].mean().reset_index()
        map_data['Year'] = 'Average'
        title_suffix = 'Trung bình (2010-2024)'
    
    if len(map_data) == 0:
        fig = go.Figure()
        fig.add_annotation(text="Không đủ dữ liệu để hiển thị biểu đồ này.",
                          xref="paper", yref="paper",
                          x=0.5, y=0.5, showarrow=False)
        return fig
    
    # Country codes for Plotly
    country_codes = {
        'Japan': 'JPN',
        'Philippines': 'PHL',
        'South Korea': 'KOR',
        'Thailand': 'THA',
        'Vietnam': 'VNM'
    }
    
    map_data['ISO_Code'] = map_data['Country'].map(country_codes)
    
    fig = go.Figure(data=go.Choropleth(
        locations=map_data['ISO_Code'],
        z=map_data['Unemployment_Rate'],
        text=map_data['Country'],
        colorscale='Blues',
        autocolorscale=False,
        reversescale=False,
        marker_line_color='darkgray',
        marker_line_width=0.5,
        colorbar_title='Tỷ lệ thất nghiệp (%)',
        hovertemplate='<b>%{text}</b><br>Tỷ lệ thất nghiệp: %{z:.2f}%<extra></extra>'
    ))
    
    fig.update_geos(
        projection_type='natural earth',
        center=dict(lon=105, lat=15),  # Center on Southeast Asia
        scope='asia'
    )
    
    fig.update_layout(
        title=f'Bản đồ tỷ lệ thất nghiệp - {title_suffix}',
        template='plotly_white',
        height=400
    )
    
    return fig


def create_gdp_unemployment_scatter(df: pd.DataFrame, countries: Optional[List[str]] = None) -> go.Figure:
    """
    Create scatter plot of GDP growth vs unemployment rate.
    
    Args:
        df: DataFrame with unemployment data
        countries: List of countries to include. If None, include all.
        
    Returns:
        Plotly figure
    """
    if countries:
        df = df[df['Country'].isin(countries)]
    
    if len(df) == 0:
        fig = go.Figure()
        fig.add_annotation(text="Không đủ dữ liệu để hiển thị biểu đồ này.",
                          xref="paper", yref="paper",
                          x=0.5, y=0.5, showarrow=False)
        return fig
    
    country_colors = {
        'Japan': '#1f77b4',
        'Philippines': '#ff7f0e',
        'South Korea': '#2ca02c',
        'Thailand': '#d62728',
        'Vietnam': '#9467bd'
    }
    
    fig = go.Figure()
    
    for country in sorted(df['Country'].unique()):
        country_data = df[df['Country'] == country]
        fig.add_trace(go.Scatter(
            x=country_data['GDP_Growth'],
            y=country_data['Unemployment_Rate'],
            mode='markers',
            name=country,
            marker=dict(
                color=country_colors.get(country, '#333333'),
                size=10,
                opacity=0.7
            ),
            hovertemplate='<b>%{fullData.name}</b><br>GDP Growth: %{x:.2f}%<br>Unemployment: %{y:.2f}%<extra></extra>'
        ))
    
    fig.update_layout(
        title='Mối liên hệ giữa tăng trưởng GDP và tỷ lệ thất nghiệp',
        xaxis_title='Tăng trưởng GDP (%)',
        yaxis_title='Tỷ lệ thất nghiệp (%)',
        hovermode='closest',
        template='plotly_white',
        height=400
    )
    
    return fig


def create_inflation_unemployment_scatter(df: pd.DataFrame, countries: Optional[List[str]] = None) -> go.Figure:
    """
    Create scatter plot of inflation vs unemployment rate.
    
    Args:
        df: DataFrame with unemployment data
        countries: List of countries to include. If None, include all.
        
    Returns:
        Plotly figure
    """
    if countries:
        df = df[df['Country'].isin(countries)]
    
    if len(df) == 0:
        fig = go.Figure()
        fig.add_annotation(text="Không đủ dữ liệu để hiển thị biểu đồ này.",
                          xref="paper", yref="paper",
                          x=0.5, y=0.5, showarrow=False)
        return fig
    
    country_colors = {
        'Japan': '#1f77b4',
        'Philippines': '#ff7f0e',
        'South Korea': '#2ca02c',
        'Thailand': '#d62728',
        'Vietnam': '#9467bd'
    }
    
    fig = go.Figure()
    
    for country in sorted(df['Country'].unique()):
        country_data = df[df['Country'] == country]
        fig.add_trace(go.Scatter(
            x=country_data['Inflation'],
            y=country_data['Unemployment_Rate'],
            mode='markers',
            name=country,
            marker=dict(
                color=country_colors.get(country, '#333333'),
                size=10,
                opacity=0.7
            ),
            hovertemplate='<b>%{fullData.name}</b><br>Inflation: %{x:.2f}%<br>Unemployment: %{y:.2f}%<extra></extra>'
        ))
    
    fig.update_layout(
        title='Mối liên hệ giữa lạm phát và tỷ lệ thất nghiệp',
        xaxis_title='Lạm phát (%)',
        yaxis_title='Tỷ lệ thất nghiệp (%)',
        hovermode='closest',
        template='plotly_white',
        height=400
    )
    
    return fig


def create_youth_unemployment_comparison(df: pd.DataFrame, countries: Optional[List[str]] = None) -> go.Figure:
    """
    Create grouped bar chart comparing youth and overall unemployment.
    
    Args:
        df: DataFrame with unemployment data
        countries: List of countries to include. If None, include all.
        
    Returns:
        Plotly figure
    """
    if countries:
        df = df[df['Country'].isin(countries)]
    
    if len(df) == 0:
        fig = go.Figure()
        fig.add_annotation(text="Không đủ dữ liệu để hiển thị biểu đồ này.",
                          xref="paper", yref="paper",
                          x=0.5, y=0.5, showarrow=False)
        return fig
    
    # Calculate average by country
    country_avg = df.groupby('Country')[['Unemployment_Rate', 'Youth_Unemployment_Rate']].mean().reset_index()
    country_avg = country_avg.sort_values('Youth_Unemployment_Rate', ascending=True)
    
    country_colors = {
        'Japan': '#1f77b4',
        'Philippines': '#ff7f0e',
        'South Korea': '#2ca02c',
        'Thailand': '#d62728',
        'Vietnam': '#9467bd'
    }
    
    fig = go.Figure()
    
    for country in country_avg['Country']:
        color = country_colors.get(country, '#333333')
        
        # Overall unemployment
        fig.add_trace(go.Bar(
            name='Tổng thể',
            x=[country],
            y=[country_avg[country_avg['Country'] == country]['Unemployment_Rate'].values[0]],
            marker_color=color,
            opacity=0.6
        ))
        
        # Youth unemployment
        fig.add_trace(go.Bar(
            name='Thanh niên (15-24)',
            x=[country],
            y=[country_avg[country_avg['Country'] == country]['Youth_Unemployment_Rate'].values[0]],
            marker_color=color,
            opacity=1.0
        ))
    
    fig.update_layout(
        barmode='group',
        title='So sánh tỷ lệ thất nghiệp: Tổng thể vs Thanh niên (2010-2024)',
        xaxis_title='Quốc gia',
        yaxis_title='Tỷ lệ thất nghiệp (%)',
        template='plotly_white',
        height=400
    )
    
    return fig


def create_period_comparison(df: pd.DataFrame, metric: str = 'Unemployment_Rate') -> go.Figure:
    """
    Create period comparison bar chart.
    
    Args:
        df: DataFrame with unemployment data
        metric: Metric to compare (default: Unemployment_Rate)
        
    Returns:
        Plotly figure
    """
    if len(df) == 0:
        fig = go.Figure()
        fig.add_annotation(text="Không đủ dữ liệu để hiển thị biểu đồ này.",
                          xref="paper", yref="paper",
                          x=0.5, y=0.5, showarrow=False)
        return fig
    
    # Add period column if not present
    if 'Period' not in df.columns:
        from src.preprocessing import add_period_column
        df = add_period_column(df)
    
    # Calculate average by period
    period_avg = df.groupby('Period')[metric].mean().reset_index()
    
    # Order periods
    period_order = ['Pre-COVID (2010-2019)', 'COVID (2020-2021)', 'Recovery (2022-2024)']
    period_avg['Period'] = pd.Categorical(period_avg['Period'], categories=period_order, ordered=True)
    period_avg = period_avg.sort_values('Period')
    
    fig = go.Figure(data=go.Bar(
        x=period_avg['Period'],
        y=period_avg[metric],
        marker_color=['#1f77b4', '#d62728', '#2ca02c'],
        text=period_avg[metric].round(2),
        textposition='outside'
    ))
    
    metric_label = 'Tỷ lệ thất nghiệp (%)' if metric == 'Unemployment_Rate' else metric
    
    fig.update_layout(
        title=f'So sánh {metric_label} theo giai đoạn',
        xaxis_title='Giai đoạn',
        yaxis_title=metric_label,
        template='plotly_white',
        height=400
    )
    
    return fig
