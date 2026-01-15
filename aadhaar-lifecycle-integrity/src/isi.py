"""
Integrity Score Index (ISI) computation module.
"""

import pandas as pd
import numpy as np
from typing import Dict, Optional
import logging

from . import config

logger = logging.getLogger(__name__)


def compute_biometric_quality_score(
    df: pd.DataFrame,
    quality_column: str = "biometric_quality_score"
) -> pd.Series:
    """
    Compute normalized biometric quality component.
    
    Parameters
    ----------
    df : pd.DataFrame
        Data with biometric quality scores.
    quality_column : str
        Name of the quality score column.
        
    Returns
    -------
    pd.Series
        Normalized quality scores (0-1).
    """
    if quality_column not in df.columns:
        logger.warning(f"Column {quality_column} not found, returning zeros")
        return pd.Series(0.0, index=df.index)
    
    # Normalize to 0-1 range
    min_score = df[quality_column].min()
    max_score = df[quality_column].max()
    
    if max_score == min_score:
        return pd.Series(1.0, index=df.index)
    
    return (df[quality_column] - min_score) / (max_score - min_score)


def compute_update_frequency_score(
    df: pd.DataFrame,
    update_count_column: str = "total_updates",
    ideal_frequency: int = 2
) -> pd.Series:
    """
    Compute update frequency component score.
    
    Higher scores for moderate update frequencies, lower for
    too few or too many updates.
    
    Parameters
    ----------
    df : pd.DataFrame
        Data with update counts.
    update_count_column : str
        Name of the update count column.
    ideal_frequency : int
        Ideal number of updates.
        
    Returns
    -------
    pd.Series
        Frequency scores (0-1).
    """
    if update_count_column not in df.columns:
        return pd.Series(0.5, index=df.index)
    
    # Use inverse of deviation from ideal
    deviation = np.abs(df[update_count_column] - ideal_frequency)
    max_deviation = deviation.max()
    
    if max_deviation == 0:
        return pd.Series(1.0, index=df.index)
    
    return 1 - (deviation / max_deviation)


def compute_consistency_score(
    biometric_updates: pd.DataFrame,
    demographic_updates: pd.DataFrame
) -> pd.DataFrame:
    """
    Compute consistency score based on update patterns.
    
    Parameters
    ----------
    biometric_updates : pd.DataFrame
        Biometric updates data.
    demographic_updates : pd.DataFrame
        Demographic updates data.
        
    Returns
    -------
    pd.DataFrame
        Consistency scores per Aadhaar number.
    """
    # Calculate quality improvement trend for biometric
    bio_quality_change = biometric_updates.groupby("aadhaar_number").apply(
        lambda x: (x["new_quality_score"] - x["previous_quality_score"]).mean()
        if len(x) > 0 else 0
    ).reset_index(name="avg_quality_change")
    
    # Normalize to 0-1 (positive change is good)
    min_change = bio_quality_change["avg_quality_change"].min()
    max_change = bio_quality_change["avg_quality_change"].max()
    
    if max_change != min_change:
        bio_quality_change["consistency_score"] = (
            bio_quality_change["avg_quality_change"] - min_change
        ) / (max_change - min_change)
    else:
        bio_quality_change["consistency_score"] = 0.5
    
    return bio_quality_change[["aadhaar_number", "consistency_score"]]


def compute_verification_score(
    df: pd.DataFrame,
    status_column: str = "verification_status",
    success_values: Optional[list] = None
) -> pd.Series:
    """
    Compute verification success rate score.
    
    Parameters
    ----------
    df : pd.DataFrame
        Data with verification status.
    status_column : str
        Name of the verification status column.
    success_values : list, optional
        Values indicating successful verification.
        
    Returns
    -------
    pd.Series
        Binary verification scores.
    """
    if success_values is None:
        success_values = ["VERIFIED", "SUCCESS", "APPROVED"]
    
    if status_column not in df.columns:
        return pd.Series(1.0, index=df.index)
    
    return df[status_column].isin(success_values).astype(float)


def compute_isi(
    enrolment_df: pd.DataFrame,
    biometric_updates: pd.DataFrame,
    demographic_updates: pd.DataFrame,
    weights: Optional[Dict[str, float]] = None
) -> pd.DataFrame:
    """
    Compute the Integrity Score Index (ISI) for each Aadhaar.
    
    ISI = w1 * BiometricQuality + w2 * UpdateFrequency + 
          w3 * Consistency + w4 * VerificationRate
    
    Parameters
    ----------
    enrolment_df : pd.DataFrame
        Enrolment data.
    biometric_updates : pd.DataFrame
        Biometric updates data.
    demographic_updates : pd.DataFrame
        Demographic updates data.
    weights : dict, optional
        Component weights. Defaults to config.ISI_WEIGHTS.
        
    Returns
    -------
    pd.DataFrame
        ISI scores per Aadhaar number.
    """
    if weights is None:
        weights = config.ISI_WEIGHTS
    
    logger.info("Computing ISI components...")
    
    # Prepare base dataframe
    result = enrolment_df[["aadhaar_number"]].copy()
    
    # Component 1: Biometric Quality
    result["quality_score"] = compute_biometric_quality_score(enrolment_df)
    
    # Component 2: Update Frequency
    bio_counts = biometric_updates.groupby("aadhaar_number").size().reset_index(name="bio_updates")
    demo_counts = demographic_updates.groupby("aadhaar_number").size().reset_index(name="demo_updates")
    
    result = result.merge(bio_counts, on="aadhaar_number", how="left")
    result = result.merge(demo_counts, on="aadhaar_number", how="left")
    result["bio_updates"] = result["bio_updates"].fillna(0)
    result["demo_updates"] = result["demo_updates"].fillna(0)
    result["total_updates"] = result["bio_updates"] + result["demo_updates"]
    result["frequency_score"] = compute_update_frequency_score(result)
    
    # Component 3: Consistency
    consistency = compute_consistency_score(biometric_updates, demographic_updates)
    result = result.merge(consistency, on="aadhaar_number", how="left")
    result["consistency_score"] = result["consistency_score"].fillna(0.5)
    
    # Component 4: Verification Rate
    result = result.merge(
        enrolment_df[["aadhaar_number", "verification_status"]],
        on="aadhaar_number",
        how="left"
    )
    result["verification_score"] = compute_verification_score(result)
    
    # Compute weighted ISI
    result["isi"] = (
        weights["biometric_quality"] * result["quality_score"] +
        weights["update_frequency"] * result["frequency_score"] +
        weights["consistency"] * result["consistency_score"] +
        weights["verification_success_rate"] * result["verification_score"]
    )
    
    logger.info(f"Computed ISI for {len(result)} Aadhaar numbers")
    logger.info(f"ISI statistics: mean={result['isi'].mean():.3f}, std={result['isi'].std():.3f}")
    
    return result


def categorize_isi(
    isi_df: pd.DataFrame,
    isi_column: str = "isi",
    thresholds: Optional[Dict[str, float]] = None
) -> pd.DataFrame:
    """
    Categorize ISI scores into integrity levels.
    
    Parameters
    ----------
    isi_df : pd.DataFrame
        DataFrame with ISI scores.
    isi_column : str
        Name of the ISI column.
    thresholds : dict, optional
        Category thresholds.
        
    Returns
    -------
    pd.DataFrame
        DataFrame with ISI category added.
    """
    if thresholds is None:
        thresholds = {
            "high": 0.8,
            "medium": 0.5,
            "low": 0.3,
        }
    
    df = isi_df.copy()
    
    conditions = [
        df[isi_column] >= thresholds["high"],
        df[isi_column] >= thresholds["medium"],
        df[isi_column] >= thresholds["low"],
    ]
    choices = ["HIGH", "MEDIUM", "LOW"]
    
    df["isi_category"] = np.select(conditions, choices, default="CRITICAL")
    
    return df
