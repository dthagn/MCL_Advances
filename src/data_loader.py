"""
Data loader module for unemployment dashboard.
Loads data from CSV files and provides access to analysis results.
"""

import pandas as pd
from pathlib import Path


def load_unemployment_data(data_path: str = "data/processed/unemployment_final.csv") -> pd.DataFrame:
    """
    Load the main unemployment dataset.
    
    Args:
        data_path: Path to the unemployment_final.csv file
        
    Returns:
        DataFrame with unemployment data
    """
    try:
        df = pd.read_csv(data_path)
        return df
    except FileNotFoundError:
        raise FileNotFoundError(f"Data file not found at {data_path}")
    except Exception as e:
        raise Exception(f"Error loading data: {str(e)}")


def load_statistics(stats_path: str = "analysis/statistics.csv") -> pd.DataFrame:
    """
    Load statistics from analysis results.
    
    Args:
        stats_path: Path to the statistics.csv file
        
    Returns:
        DataFrame with statistics
    """
    try:
        df = pd.read_csv(stats_path)
        return df
    except FileNotFoundError:
        raise FileNotFoundError(f"Statistics file not found at {stats_path}")
    except Exception as e:
        raise Exception(f"Error loading statistics: {str(e)}")


def load_analysis_results(results_path: str = "analysis/analysis_results.csv") -> pd.DataFrame:
    """
    Load analysis results.
    
    Args:
        results_path: Path to the analysis_results.csv file
        
    Returns:
        DataFrame with analysis results
    """
    try:
        df = pd.read_csv(results_path)
        return df
    except FileNotFoundError:
        raise FileNotFoundError(f"Analysis results file not found at {results_path}")
    except Exception as e:
        raise Exception(f"Error loading analysis results: {str(e)}")


def load_correlation_results(correlation_path: str = "analysis/correlation_results.csv") -> pd.DataFrame:
    """
    Load correlation results.
    
    Args:
        correlation_path: Path to the correlation_results.csv file
        
    Returns:
        DataFrame with correlation results
    """
    try:
        df = pd.read_csv(correlation_path)
        return df
    except FileNotFoundError:
        raise FileNotFoundError(f"Correlation results file not found at {correlation_path}")
    except Exception as e:
        raise Exception(f"Error loading correlation results: {str(e)}")


def load_findings(findings_path: str = "analysis/findings.md") -> str:
    """
    Load findings from markdown file.
    
    Args:
        findings_path: Path to the findings.md file
        
    Returns:
        String with findings content
    """
    try:
        with open(findings_path, 'r', encoding='utf-8') as f:
            content = f.read()
        return content
    except FileNotFoundError:
        raise FileNotFoundError(f"Findings file not found at {findings_path}")
    except Exception as e:
        raise Exception(f"Error loading findings: {str(e)}")


def load_data_dictionary(dict_path: str = "data/processed/data_dictionary.csv") -> pd.DataFrame:
    """
    Load data dictionary.
    
    Args:
        dict_path: Path to the data_dictionary.csv file
        
    Returns:
        DataFrame with data dictionary
    """
    try:
        df = pd.read_csv(dict_path)
        return df
    except FileNotFoundError:
        raise FileNotFoundError(f"Data dictionary file not found at {dict_path}")
    except Exception as e:
        raise Exception(f"Error loading data dictionary: {str(e)}")


def load_source_metadata(metadata_path: str = "data/processed/source_metadata.csv") -> pd.DataFrame:
    """
    Load source metadata.
    
    Args:
        metadata_path: Path to the source_metadata.csv file
        
    Returns:
        DataFrame with source metadata
    """
    try:
        df = pd.read_csv(metadata_path)
        return df
    except FileNotFoundError:
        raise FileNotFoundError(f"Source metadata file not found at {metadata_path}")
    except Exception as e:
        raise Exception(f"Error loading source metadata: {str(e)}")
