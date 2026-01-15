"""
Data Update Velocity (DUV) computation module.
"""

import pandas as pd
import numpy as np
from typing import Optional, Tuple
import logging

from . import config

logger = logging.getLogger(__name__)


def compute_update_velocity(
    updates_df: pd.DataFrame,
    date_column: str = "update_date",
    window_days: int = None
) -> pd.DataFrame:
    """
    Compute update velocity (updates per time period).
    
    Parameters
    ----------
    updates_df : pd.DataFrame
        Updates data.
    date_column : str
        Name of the date column.
    window_days : int, optional
        Rolling window size in days.
        
    Returns
    -------
    pd.DataFrame
        Velocity metrics per Aadhaar number.
    """
    if window_days is None:
        window_days = config.DUV_WINDOW_DAYS
    
    df = updates_df.copy()
    df = df.sort_values(date_column)
    
    # Calculate time range per Aadhaar
    velocity = df.groupby("aadhaar_number").agg({
        date_column: ["min", "max", "count"]
    })
    velocity.columns = ["first_update", "last_update", "update_count"]
    velocity = velocity.reset_index()
    
    # Calculate time span
    velocity["time_span_days"] = (
        velocity["last_update"] - velocity["first_update"]
    ).dt.days + 1  # Add 1 to include both endpoints
    
    # Calculate velocity (updates per year)
    velocity["duv"] = (
        velocity["update_count"] / velocity["time_span_days"]
    ) * 365
    
    # Handle single updates (set to normalized count)
    velocity.loc[velocity["time_span_days"] == 1, "duv"] = velocity["update_count"]
    
    return velocity


def compute_acceleration(
    updates_df: pd.DataFrame,
    date_column: str = "update_date"
) -> pd.DataFrame:
    """
    Compute update acceleration (change in velocity over time).
    
    Parameters
    ----------
    updates_df : pd.DataFrame
        Updates data.
    date_column : str
        Name of the date column.
        
    Returns
    -------
    pd.DataFrame
        Acceleration metrics per Aadhaar number.
    """
    df = updates_df.copy()
    df = df.sort_values([date_column])
    
    # Calculate inter-update intervals
    df["prev_update"] = df.groupby("aadhaar_number")[date_column].shift(1)
    df["interval_days"] = (df[date_column] - df["prev_update"]).dt.days
    
    # Calculate acceleration per Aadhaar
    acceleration = df.groupby("aadhaar_number").agg({
        "interval_days": ["mean", "std", lambda x: x.diff().mean()]
    })
    acceleration.columns = ["mean_interval", "std_interval", "acceleration"]
    acceleration = acceleration.reset_index()
    
    # Negative acceleration = intervals getting smaller = speeding up
    acceleration["acceleration"] = -acceleration["acceleration"].fillna(0)
    
    return acceleration


def compute_rolling_velocity(
    updates_df: pd.DataFrame,
    date_column: str = "update_date",
    window: str = "30D"
) -> pd.DataFrame:
    """
    Compute rolling velocity over time.
    
    Parameters
    ----------
    updates_df : pd.DataFrame
        Updates data.
    date_column : str
        Name of the date column.
    window : str
        Rolling window size (e.g., '30D', '7D').
        
    Returns
    -------
    pd.DataFrame
        Time series of rolling velocity.
    """
    df = updates_df.copy()
    df = df.set_index(date_column)
    
    # Count updates per day
    daily_counts = df.groupby(df.index.date).size()
    daily_counts = pd.DataFrame({"count": daily_counts})
    daily_counts.index = pd.to_datetime(daily_counts.index)
    
    # Calculate rolling sum
    daily_counts["rolling_velocity"] = daily_counts["count"].rolling(
        window=window,
        min_periods=1
    ).mean()
    
    return daily_counts.reset_index().rename(columns={"index": "date"})


def compute_duv_by_modality(
    biometric_updates: pd.DataFrame
) -> pd.DataFrame:
    """
    Compute DUV broken down by biometric modality.
    
    Parameters
    ----------
    biometric_updates : pd.DataFrame
        Biometric updates data.
        
    Returns
    -------
    pd.DataFrame
        DUV by modality.
    """
    if "biometric_modality" not in biometric_updates.columns:
        logger.warning("No biometric_modality column found")
        return pd.DataFrame()
    
    modality_counts = biometric_updates.groupby(
        ["aadhaar_number", "biometric_modality"]
    ).size().unstack(fill_value=0)
    
    # Calculate velocities per modality
    modality_counts = modality_counts.add_suffix("_count")
    
    return modality_counts.reset_index()


def compute_duv_score(
    velocity_df: pd.DataFrame,
    duv_column: str = "duv",
    optimal_range: Tuple[float, float] = (0.5, 2.0)
) -> pd.DataFrame:
    """
    Convert raw DUV to normalized score.
    
    Scores indicate deviation from optimal update velocity.
    
    Parameters
    ----------
    velocity_df : pd.DataFrame
        Velocity data.
    duv_column : str
        Name of the DUV column.
    optimal_range : tuple
        Optimal velocity range (min, max).
        
    Returns
    -------
    pd.DataFrame
        DUV with normalized score.
    """
    df = velocity_df.copy()
    
    lower, upper = optimal_range
    
    # Score = 1 if within optimal range, decreases as you move away
    conditions = [
        (df[duv_column] >= lower) & (df[duv_column] <= upper),
        df[duv_column] < lower,
        df[duv_column] > upper,
    ]
    
    # Score calculations
    choices = [
        1.0,  # Optimal range
        df[duv_column] / lower,  # Below optimal (scale up)
        upper / df[duv_column],  # Above optimal (scale down)
    ]
    
    df["duv_score"] = np.select(conditions, choices, default=0)
    df["duv_score"] = df["duv_score"].clip(0, 1)
    
    return df


def categorize_duv(
    df: pd.DataFrame,
    duv_column: str = "duv"
) -> pd.DataFrame:
    """
    Categorize DUV into velocity categories.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with DUV values.
    duv_column : str
        Name of the DUV column.
        
    Returns
    -------
    pd.DataFrame
        DataFrame with DUV category.
    """
    df = df.copy()
    
    conditions = [
        df[duv_column] == 0,
        df[duv_column] < 0.5,
        df[duv_column] < 2.0,
        df[duv_column] < 5.0,
    ]
    choices = ["DORMANT", "LOW", "NORMAL", "HIGH"]
    
    df["duv_category"] = np.select(conditions, choices, default="HYPERACTIVE")
    
    return df
