"""
Cohort Lag Alignment Module for Aadhaar Lifecycle Integrity Analysis.

Core Principle:
--------------
Children enrolled at age 0-5 in year (t-5) are expected to require
Mandatory Biometric Updates (MBU) when they reach age 5-10 in year (t).

This module aligns enrolment cohorts with update observations using
fixed temporal lags to compute:
- Expected updates: Based on historical enrolments
- Observed updates: Actual biometric update activity
- Update gap: Difference indicating maintenance shortfall

Design Considerations:
---------------------
1. The data currently available (2025) represents a single year.
   Multi-year analysis requires historical enrolment data from 2020
   to properly compute 5-year lag expectations for 2025 updates.

2. When multi-year data is unavailable, this module provides:
   - Within-year cohort structure (using monthly granularity if available)
   - District-level expected vs observed comparison
   - Geographic aggregation support ready for multi-year analysis

3. The module is designed to be deterministic and reproducible.

Author: Principal Data Scientist
Constraints: Government-grade, policy-safe, deterministic
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Literal
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


# =============================================================================
# CONFIGURATION: LAG STRUCTURE
# =============================================================================

# Biometric update lag: Children enrolled at age 0-5 need updates at age 5-10
# This means enrolments from year (t-5) predict updates in year (t)
BIOMETRIC_LAG_YEARS = 5

# Age groups that predict future mandatory biometric updates
# Age 0-5 enrolments -> Updates in 5 years
# Age 5-17 may need periodic updates (every 10 years per UIDAI policy)
CHILD_ENROLMENT_TO_UPDATE_LAG = {
    "age_0_5": 5,       # 0-5 year olds -> update at age 5-10
    "age_5_17": 10,     # 5-17 year olds -> update at age 15-27
}

# Aggregation level options
AGGREGATION_LEVELS = Literal["pincode", "district", "state"]


@dataclass
class CohortAlignmentResult:
    """Container for cohort alignment results."""
    data: pd.DataFrame
    metadata: Dict
    
    def __str__(self) -> str:
        return (
            f"CohortAlignmentResult(rows={len(self.data)}, "
            f"cohort_years={self.metadata.get('cohort_years', 'N/A')})"
        )


# =============================================================================
# LAG ALIGNMENT FUNCTIONS
# =============================================================================

def compute_expected_updates(
    enrolment_df: pd.DataFrame,
    target_year: int,
    lag_years: int = BIOMETRIC_LAG_YEARS,
    source_age_column: str = "age_0_5",
    aggregation_level: AGGREGATION_LEVELS = "pincode"
) -> pd.DataFrame:
    """
    Compute expected biometric updates for a target year based on lagged enrolments.
    
    Reasoning:
    ---------
    If a child was enrolled at age 0-5 in year (target_year - lag_years),
    they would be age (5 to 10) in target_year and may require MBU.
    
    The expected update count for a geography is the sum of age_0_5
    enrolments from lag_years ago in that geography.
    
    Assumptions:
    -----------
    1. All children enrolled at age 0-5 will eventually need MBU at age 5+
    2. Geographic residence is stable (no migration adjustment)
    3. Lag is fixed at 5 years (UIDAI child-to-adult biometric transition)
    4. 100% expected update rate (conservative upper bound)
    
    Parameters
    ----------
    enrolment_df : pd.DataFrame
        Yearly aggregated enrolment data with 'year' column.
    target_year : int
        Year for which to compute expected updates.
    lag_years : int
        Number of years to look back for source enrolments.
    source_age_column : str
        Column containing the enrolment counts to use as expectation source.
    aggregation_level : str
        Geographic level: 'pincode', 'district', or 'state'.
        
    Returns
    -------
    pd.DataFrame
        Expected updates by geography with columns:
        - geographic identifiers (state, district, optionally pincode)
        - cohort_year: The year of original enrolment (source year)
        - target_year: The year updates are expected
        - expected_updates: Count of expected updates
    """
    # Compute source year (when the cohort was enrolled)
    source_year = target_year - lag_years
    
    # Filter to source year enrolments
    source_data = enrolment_df[enrolment_df["year"] == source_year].copy()
    
    if len(source_data) == 0:
        logger.warning(
            f"No enrolment data for source year {source_year}. "
            f"Cannot compute expectations for target year {target_year}."
        )
        return pd.DataFrame()
    
    # Define grouping columns based on aggregation level
    if aggregation_level == "pincode":
        group_cols = ["state", "district", "pincode"]
    elif aggregation_level == "district":
        group_cols = ["state", "district"]
    else:  # state
        group_cols = ["state"]
    
    # Aggregate expected updates
    expected = source_data.groupby(group_cols, as_index=False).agg({
        source_age_column: "sum"
    })
    
    # Rename and add metadata columns
    expected = expected.rename(columns={source_age_column: "expected_updates"})
    expected["cohort_year"] = source_year
    expected["target_year"] = target_year
    expected["lag_years"] = lag_years
    expected["source_column"] = source_age_column
    
    logger.info(
        f"Computed expected updates for {target_year}: "
        f"{expected['expected_updates'].sum():,} total from {len(expected)} geographies"
    )
    
    return expected


def compute_observed_updates(
    biometric_df: pd.DataFrame,
    target_year: int,
    aggregation_level: AGGREGATION_LEVELS = "pincode"
) -> pd.DataFrame:
    """
    Compute observed biometric updates for a target year.
    
    Reasoning:
    ---------
    Sum all biometric update counts in the target year by geography.
    This represents the actual maintenance activity observed.
    
    Assumptions:
    -----------
    1. All biometric update records represent valid maintenance activity
    2. Duplicates have been removed in preprocessing
    3. Both age groups (5-17 and 17+) contribute to observed updates
    
    Parameters
    ----------
    biometric_df : pd.DataFrame
        Yearly aggregated biometric update data.
    target_year : int
        Year for which to compute observed updates.
    aggregation_level : str
        Geographic level: 'pincode', 'district', or 'state'.
        
    Returns
    -------
    pd.DataFrame
        Observed updates by geography.
    """
    # Filter to target year
    target_data = biometric_df[biometric_df["year"] == target_year].copy()
    
    if len(target_data) == 0:
        logger.warning(f"No biometric update data for year {target_year}.")
        return pd.DataFrame()
    
    # Define grouping columns
    if aggregation_level == "pincode":
        group_cols = ["state", "district", "pincode"]
    elif aggregation_level == "district":
        group_cols = ["state", "district"]
    else:  # state
        group_cols = ["state"]
    
    # Aggregate observed updates (sum both age group columns if available)
    agg_dict = {}
    for col in ["total_bio_updates", "bio_age_5_17", "bio_age_17_"]:
        if col in target_data.columns:
            agg_dict[col] = "sum"
    
    observed = target_data.groupby(group_cols, as_index=False).agg(agg_dict)
    
    # Use total_bio_updates as the primary observed count
    if "total_bio_updates" in observed.columns:
        observed = observed.rename(columns={"total_bio_updates": "observed_updates"})
    elif "bio_age_5_17" in observed.columns and "bio_age_17_" in observed.columns:
        observed["observed_updates"] = observed["bio_age_5_17"] + observed["bio_age_17_"]
    
    observed["observation_year"] = target_year
    
    logger.info(
        f"Computed observed updates for {target_year}: "
        f"{observed['observed_updates'].sum():,} total from {len(observed)} geographies"
    )
    
    return observed


def align_cohorts(
    enrolment_df: pd.DataFrame,
    biometric_df: pd.DataFrame,
    target_year: int,
    lag_years: int = BIOMETRIC_LAG_YEARS,
    aggregation_level: AGGREGATION_LEVELS = "pincode"
) -> pd.DataFrame:
    """
    Align enrolment cohorts with observed updates using temporal lag.
    
    Core Operation:
    --------------
    1. Compute expected updates from enrolments (target_year - lag_years)
    2. Compute observed updates in target_year
    3. Join on geography
    4. Compute update_gap = expected - observed
    
    The update_gap represents the "silent failure" in biometric maintenance:
    - Positive gap: Fewer updates than expected (under-maintenance)
    - Negative gap: More updates than expected (could indicate catch-up or data issues)
    - Zero gap: Perfect alignment (rare in practice)
    
    Assumptions:
    -----------
    1. Geographic keys are consistent between enrolment and update data
    2. Expected updates use age_0_5 enrolments from lag years ago
    3. Observed updates include all biometric update activity
    4. Migration and mortality are not modeled (limitation)
    
    Parameters
    ----------
    enrolment_df : pd.DataFrame
        Yearly aggregated enrolment data.
    biometric_df : pd.DataFrame
        Yearly aggregated biometric update data.
    target_year : int
        Year for which to perform alignment.
    lag_years : int
        Years of lag between enrolment and expected update.
    aggregation_level : str
        Geographic level for aggregation.
        
    Returns
    -------
    pd.DataFrame
        Aligned cohort data with columns:
        - geography columns (state, district, [pincode])
        - cohort_year: Year of original enrolment
        - target_year: Year of expected updates
        - expected_updates: Count based on lagged enrolments
        - observed_updates: Actual updates observed
        - update_gap: expected - observed (positive = under-maintenance)
        - update_ratio: observed / expected (< 1 = under-maintenance)
    """
    # Compute expected and observed
    expected = compute_expected_updates(
        enrolment_df, target_year, lag_years, "age_0_5", aggregation_level
    )
    observed = compute_observed_updates(
        biometric_df, target_year, aggregation_level
    )
    
    if len(expected) == 0 or len(observed) == 0:
        logger.warning(
            f"Cannot align cohorts: expected={len(expected)}, observed={len(observed)}"
        )
        return pd.DataFrame()
    
    # Define join keys
    if aggregation_level == "pincode":
        join_keys = ["state", "district", "pincode"]
    elif aggregation_level == "district":
        join_keys = ["state", "district"]
    else:
        join_keys = ["state"]
    
    # Merge expected and observed
    # Use outer join to capture geographies with only expected OR only observed
    aligned = expected.merge(
        observed[join_keys + ["observed_updates"]],
        on=join_keys,
        how="outer"
    )
    
    # Fill NaN with 0 for proper gap calculation
    # NaN in expected means no historical enrolments (unexpected updates)
    # NaN in observed means no updates occurred (potential staleness)
    aligned["expected_updates"] = aligned["expected_updates"].fillna(0)
    aligned["observed_updates"] = aligned["observed_updates"].fillna(0)
    
    # Compute update gap and ratio
    aligned["update_gap"] = aligned["expected_updates"] - aligned["observed_updates"]
    
    # Safe ratio calculation (avoid division by zero)
    aligned["update_ratio"] = np.where(
        aligned["expected_updates"] > 0,
        aligned["observed_updates"] / aligned["expected_updates"],
        np.nan  # No expected updates = undefined ratio
    )
    
    # Fill cohort info for rows that came from observed-only
    aligned["cohort_year"] = aligned["cohort_year"].fillna(target_year - lag_years)
    aligned["target_year"] = aligned["target_year"].fillna(target_year)
    aligned["lag_years"] = aligned["lag_years"].fillna(lag_years)
    
    logger.info(
        f"Cohort alignment complete: {len(aligned)} geographies, "
        f"total gap = {aligned['update_gap'].sum():,.0f}"
    )
    
    return aligned


# =============================================================================
# MULTI-LEVEL AGGREGATION
# =============================================================================

def aggregate_alignment_to_district(
    aligned_df: pd.DataFrame
) -> pd.DataFrame:
    """
    Aggregate PIN-level alignment data to district level.
    
    Useful for reducing noise in sparse PIN-level data while
    maintaining district-level integrity analysis.
    
    Parameters
    ----------
    aligned_df : pd.DataFrame
        PIN-level aligned cohort data.
        
    Returns
    -------
    pd.DataFrame
        District-level aggregated data.
    """
    if "pincode" not in aligned_df.columns:
        logger.info("Data already at district level or higher")
        return aligned_df
    
    agg_dict = {
        "expected_updates": "sum",
        "observed_updates": "sum",
        "update_gap": "sum",
    }
    
    # Preserve first value for metadata columns
    meta_cols = ["cohort_year", "target_year", "lag_years"]
    for col in meta_cols:
        if col in aligned_df.columns:
            agg_dict[col] = "first"
    
    district_agg = aligned_df.groupby(
        ["state", "district"], as_index=False
    ).agg(agg_dict)
    
    # Recompute ratio at aggregated level
    district_agg["update_ratio"] = np.where(
        district_agg["expected_updates"] > 0,
        district_agg["observed_updates"] / district_agg["expected_updates"],
        np.nan
    )
    
    # Add pincode count for reference
    pincode_counts = aligned_df.groupby(
        ["state", "district"]
    )["pincode"].nunique().reset_index(name="pincode_count")
    
    district_agg = district_agg.merge(pincode_counts, on=["state", "district"])
    
    logger.info(
        f"Aggregated to district level: {len(district_agg)} districts "
        f"from {len(aligned_df)} pincodes"
    )
    
    return district_agg


def aggregate_alignment_to_state(
    aligned_df: pd.DataFrame
) -> pd.DataFrame:
    """
    Aggregate alignment data to state level.
    
    Parameters
    ----------
    aligned_df : pd.DataFrame
        PIN or district level aligned data.
        
    Returns
    -------
    pd.DataFrame
        State-level aggregated data.
    """
    agg_dict = {
        "expected_updates": "sum",
        "observed_updates": "sum",
        "update_gap": "sum",
    }
    
    meta_cols = ["cohort_year", "target_year", "lag_years"]
    for col in meta_cols:
        if col in aligned_df.columns:
            agg_dict[col] = "first"
    
    state_agg = aligned_df.groupby(["state"], as_index=False).agg(agg_dict)
    
    # Recompute ratio
    state_agg["update_ratio"] = np.where(
        state_agg["expected_updates"] > 0,
        state_agg["observed_updates"] / state_agg["expected_updates"],
        np.nan
    )
    
    logger.info(
        f"Aggregated to state level: {len(state_agg)} states"
    )
    
    return state_agg


# =============================================================================
# SINGLE-YEAR DATA HANDLING
# =============================================================================

def compute_same_year_cohort_ratio(
    enrolment_df: pd.DataFrame,
    biometric_df: pd.DataFrame,
    year: int,
    aggregation_level: AGGREGATION_LEVELS = "district"
) -> pd.DataFrame:
    """
    Compute update-to-enrolment ratio for same-year data.
    
    When multi-year lagged data is unavailable, this function provides
    a proxy metric: the ratio of biometric updates to total enrolments
    in the same year.
    
    Interpretation:
    --------------
    - This is NOT the same as the lagged cohort alignment
    - It measures "update activity relative to enrolment activity"
    - Higher ratio may indicate catch-up maintenance or older population
    - Lower ratio may indicate newer population or under-maintenance
    
    Limitations:
    -----------
    1. Does not capture true cohort staleness
    2. Conflates new enrolments with update activity
    3. Should only be used when lagged data is unavailable
    
    Parameters
    ----------
    enrolment_df : pd.DataFrame
        Yearly enrolment data.
    biometric_df : pd.DataFrame
        Yearly biometric update data.
    year : int
        Year to analyze.
    aggregation_level : str
        Geographic aggregation level.
        
    Returns
    -------
    pd.DataFrame
        Same-year ratio analysis with columns:
        - geography columns
        - total_enrolments, total_bio_updates
        - same_year_ratio: updates / enrolments
    """
    # Define group columns
    if aggregation_level == "pincode":
        group_cols = ["state", "district", "pincode"]
    elif aggregation_level == "district":
        group_cols = ["state", "district"]
    else:
        group_cols = ["state"]
    
    # Filter and aggregate enrolments
    enrol_year = enrolment_df[enrolment_df["year"] == year].copy()
    enrol_agg = enrol_year.groupby(group_cols, as_index=False).agg({
        "total_enrolments": "sum"
    })
    
    # Filter and aggregate biometric updates
    bio_year = biometric_df[biometric_df["year"] == year].copy()
    bio_agg = bio_year.groupby(group_cols, as_index=False).agg({
        "total_bio_updates": "sum"
    })
    
    # Merge
    result = enrol_agg.merge(bio_agg, on=group_cols, how="outer")
    result = result.fillna(0)
    
    # Compute ratio
    result["same_year_ratio"] = np.where(
        result["total_enrolments"] > 0,
        result["total_bio_updates"] / result["total_enrolments"],
        np.nan
    )
    
    result["year"] = year
    result["metric_type"] = "same_year_proxy"
    
    logger.info(
        f"Same-year ratio computed for {year}: "
        f"mean ratio = {result['same_year_ratio'].mean():.3f}"
    )
    
    return result


# =============================================================================
# MAIN ALIGNMENT PIPELINE
# =============================================================================

def perform_cohort_alignment(
    enrolment_df: pd.DataFrame,
    biometric_df: pd.DataFrame,
    target_years: Optional[List[int]] = None,
    lag_years: int = BIOMETRIC_LAG_YEARS,
    aggregation_level: AGGREGATION_LEVELS = "district",
    fallback_same_year: bool = True
) -> CohortAlignmentResult:
    """
    Main pipeline for cohort lag alignment.
    
    This function orchestrates the alignment process:
    1. Determines available years in the data
    2. For each target year, attempts lagged cohort alignment
    3. Falls back to same-year ratio if lagged data unavailable
    4. Returns consolidated results with metadata
    
    Parameters
    ----------
    enrolment_df : pd.DataFrame
        Preprocessed yearly enrolment data.
    biometric_df : pd.DataFrame
        Preprocessed yearly biometric update data.
    target_years : list, optional
        Years to perform alignment for. If None, uses all years
        in biometric data.
    lag_years : int
        Lag years for cohort alignment.
    aggregation_level : str
        Geographic level for analysis.
    fallback_same_year : bool
        Whether to use same-year ratio when lagged data unavailable.
        
    Returns
    -------
    CohortAlignmentResult
        Alignment results with data and metadata.
    """
    logger.info("Starting cohort alignment pipeline...")
    
    # Determine target years
    if target_years is None:
        target_years = sorted(biometric_df["year"].unique())
    
    available_enrol_years = set(enrolment_df["year"].unique())
    
    # Track results
    aligned_dfs = []
    fallback_dfs = []
    alignments_performed = 0
    fallbacks_performed = 0
    
    for target_year in target_years:
        source_year = target_year - lag_years
        
        if source_year in available_enrol_years:
            # Perform proper lagged alignment
            aligned = align_cohorts(
                enrolment_df, biometric_df, target_year,
                lag_years, aggregation_level
            )
            if len(aligned) > 0:
                aligned_dfs.append(aligned)
                alignments_performed += 1
        elif fallback_same_year:
            # Use same-year proxy
            logger.warning(
                f"Source year {source_year} not available for target {target_year}. "
                f"Using same-year ratio as fallback."
            )
            fallback = compute_same_year_cohort_ratio(
                enrolment_df, biometric_df, target_year, aggregation_level
            )
            if len(fallback) > 0:
                fallback_dfs.append(fallback)
                fallbacks_performed += 1
        else:
            logger.warning(
                f"Skipping target year {target_year}: "
                f"source year {source_year} not available and fallback disabled."
            )
    
    # Consolidate results
    if aligned_dfs:
        result_df = pd.concat(aligned_dfs, ignore_index=True)
        result_type = "lagged_cohort"
    elif fallback_dfs:
        result_df = pd.concat(fallback_dfs, ignore_index=True)
        result_type = "same_year_proxy"
    else:
        result_df = pd.DataFrame()
        result_type = "none"
    
    metadata = {
        "target_years": target_years,
        "lag_years": lag_years,
        "aggregation_level": aggregation_level,
        "alignments_performed": alignments_performed,
        "fallbacks_performed": fallbacks_performed,
        "result_type": result_type,
        "available_enrolment_years": list(available_enrol_years),
    }
    
    logger.info(
        f"Cohort alignment complete: "
        f"{alignments_performed} lagged, {fallbacks_performed} fallback"
    )
    
    return CohortAlignmentResult(data=result_df, metadata=metadata)
