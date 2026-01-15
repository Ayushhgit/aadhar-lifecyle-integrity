"""
Cohort Flow Consistency Alignment Module.

Core Principle:
--------------
Cross-Sectional Cohort Flow Consistency (2025 Data).

In a demographically stable system, "Entry Enrolments" (Age 0-5) represent
localized "Identity Entry Pressure". Mandatory Biometric Update (MBU) demand
(Age 5-17) should scale proportionately with this entry pressure.

This module aligns functional flows within the same administrative year to
evaluate system throughput consistency.

Metrics:
-------
- Expected Updates: Proportional to Entry Enrolments (0-5)
- Observed Updates: Actual Biometric Updates (5-17)
- Flow Consistency: Observed / Expected

Design Constraints:
------------------
- Strict cross-sectional analysis (2025 only)
- No historical cohort aging assumed
- No individual-level inference

Author: Principal Data Scientist
Constraints: Government-grade, policy-safe, deterministic
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Literal, Tuple
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


# =============================================================================
# CONFIGURATION
# =============================================================================

# Structural Normalization Constant (phi)
# Default 1.0 implies magnitude comparison (1:1 scaling)
DEFAULT_PHI = 1.0

# Aggregation Levels
AGGREGATION_LEVELS = Literal["pincode", "district", "state"]


@dataclass
class CohortAlignmentResult:
    """Container for cohort alignment results."""
    data: pd.DataFrame
    metadata: Dict
    
    def __str__(self) -> str:
        return (
            f"CohortAlignmentResult(rows={len(self.data)}, "
            f"mode={self.metadata.get('mode', 'unknown')})"
        )


# =============================================================================
# CROSS-SECTIONAL EXPECTATION LOGIC
# =============================================================================

def compute_cross_sectional_expectations(
    enrolment_df: pd.DataFrame,
    target_year: int,
    phi: float = DEFAULT_PHI,
    source_age_column: str = "age_0_5",
    aggregation_level: AGGREGATION_LEVELS = "pincode"
) -> pd.DataFrame:
    """
    Compute expected updates based on Cross-Sectional Cohort Flow Consistency.
    
    Logic:
    -----
    Expected Updates (5-17) = Enrolments (0-5) * phi
    
    This treats Age 0-5 enrolments as the proxy for "Identity Entry Pressure".
    
    Parameters
    ----------
    enrolment_df : pd.DataFrame
        Yearly aggregated enrolment data.
    target_year : int
        The year to analyze (e.g., 2025).
    phi : float
        Structural normalization constant.
    source_age_column : str
        Column representing entry pressure (default: age_0_5).
    aggregation_level : str
        Geographic level: 'pincode', 'district', or 'state'.
        
    Returns
    -------
    pd.DataFrame
        DataFrame with 'expected_updates' and geographic keys.
    """
    # Filter to target year (Same-Year Flow)
    source_data = enrolment_df[enrolment_df["year"] == target_year].copy()
    
    if len(source_data) == 0:
        logger.warning(f"No enrolment data for year {target_year}.")
        return pd.DataFrame()
    
    # Define grouping
    if aggregation_level == "pincode":
        group_cols = ["state", "district", "pincode"]
    elif aggregation_level == "district":
        group_cols = ["state", "district"]
    else:
        group_cols = ["state"]
    
    # Aggregate source flow
    expected = source_data.groupby(group_cols, as_index=False).agg({
        source_age_column: "sum"
    })
    
    # Apply structural model
    expected["expected_updates"] = expected[source_age_column] * phi
    
    # Metadata columns
    expected["year"] = target_year
    expected["expectation_model"] = "cross_sectional_flow"
    expected["phi"] = phi
    
    logger.info(
        f"Computed cross-sectional expectations for {target_year}: "
        f"sum={expected['expected_updates'].sum():,.0f} (phi={phi})"
    )
    
    return expected


def compute_observed_updates(
    biometric_df: pd.DataFrame,
    target_year: int,
    aggregation_level: AGGREGATION_LEVELS = "pincode"
) -> pd.DataFrame:
    """
    Compute observed biometric updates (Age 5-17 + 17+) for target year.
    
    Parameters
    ----------
    biometric_df : pd.DataFrame
        Yearly aggregated biometric update data.
    target_year : int
        Year to analyze.
    aggregation_level : str
        Geographic level.
        
    Returns
    -------
    pd.DataFrame
        DataFrame with 'observed_updates' (total) and components.
    """
    # Filter to target year
    target_data = biometric_df[biometric_df["year"] == target_year].copy()
    
    if len(target_data) == 0:
        logger.warning(f"No biometric update data for year {target_year}.")
        return pd.DataFrame()
    
    # Define grouping
    if aggregation_level == "pincode":
        group_cols = ["state", "district", "pincode"]
    elif aggregation_level == "district":
        group_cols = ["state", "district"]
    else:
        group_cols = ["state"]
    
    # Aggregate observed components
    agg_dict = {
        "total_bio_updates": "sum",
        "bio_age_5_17": "sum",
        "bio_age_17_": "sum"
    }
    # Only aggregate columns that exist
    agg_dict = {k: v for k, v in agg_dict.items() if k in target_data.columns}
    
    observed = target_data.groupby(group_cols, as_index=False).agg(agg_dict)
    
    # Standardize output column
    if "total_bio_updates" in observed.columns:
        observed = observed.rename(columns={"total_bio_updates": "observed_updates"})
    else:
        # Fallback if total not pre-calculated
        observed["observed_updates"] = (
            observed.get("bio_age_5_17", 0) + observed.get("bio_age_17_", 0)
        )
        
    observed["year"] = target_year
    
    return observed


# =============================================================================
# ALIGNMENT PIPELINE
# =============================================================================

def perform_cohort_alignment(
    enrolment_df: pd.DataFrame,
    biometric_df: pd.DataFrame,
    target_year: int = 2025,
    phi: float = DEFAULT_PHI,
    aggregation_level: AGGREGATION_LEVELS = "district"
) -> CohortAlignmentResult:
    """
    Execute Cohort Flow Consistency Alignment.
    
    Aligns Entry Enrolment Pressure (0-5) with Biometric Update Volume (5-17).
    
    Parameters
    ----------
    enrolment_df : pd.DataFrame
        Preprocessed enrolment data.
    biometric_df : pd.DataFrame
        Preprocessed biometric data.
    target_year : int
        Year of analysis (default 2025).
    phi : float
        Structural constant.
    aggregation_level : str
        Geographic level (recommend 'district' for stability).
        
    Returns
    -------
    CohortAlignmentResult
        Aligned data containing expected and observed flows.
    """
    logger.info(f"Starting Cohort Flow Alignment for {target_year}...")
    
    # 1. Compute Expectations (from Enrolments)
    expected = compute_cross_sectional_expectations(
        enrolment_df, target_year, phi, "age_0_5", aggregation_level
    )
    
    # 2. Compute Observed (from Updates)
    observed = compute_observed_updates(
        biometric_df, target_year, aggregation_level
    )
    
    if len(expected) == 0 or len(observed) == 0:
        logger.error("Failed to compute flows. Check data availability.")
        return CohortAlignmentResult(pd.DataFrame(), {"error": "no_data"})
    
    # 3. Align (Merge)
    if aggregation_level == "pincode":
        join_keys = ["state", "district", "pincode"]
    elif aggregation_level == "district":
        join_keys = ["state", "district"]
    else:
        join_keys = ["state"]
        
    aligned = expected.merge(
        observed.drop(columns=["year"]), 
        on=join_keys, 
        how="outer"
    )
    
    # Fill gaps (structurally absent flows = 0)
    aligned["expected_updates"] = aligned["expected_updates"].fillna(0)
    aligned["observed_updates"] = aligned["observed_updates"].fillna(0)
    
    # Metadata
    metadata = {
        "mode": "cross_sectional_flow_consistency",
        "year": target_year,
        "phi": phi,
        "aggregation": aggregation_level,
        "rows": len(aligned)
    }
    
    logger.info(f"Alignment complete. {len(aligned)} geographic units aligned.")
    
    return CohortAlignmentResult(aligned, metadata)
