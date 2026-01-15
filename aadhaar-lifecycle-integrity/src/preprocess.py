"""
Preprocessing Functions for UIDAI Aggregated Datasets.

This module provides functions to transform raw loaded data into
analytics-ready DataFrames with:
- Standardized temporal granularity (yearly aggregation)
- Population-relative normalization
- Geographic consistency across datasets
- Explicit handling of sparse PIN-level data

Design Principles:
- Correctness over performance
- Interpretability over cleverness
- Reproducibility through explicit documentation

Author: Principal Data Scientist
Constraints: Government-grade, policy-safe, deterministic
"""

import pandas as pd
import numpy as np
from typing import Dict, Tuple, Optional, List
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


# =============================================================================
# CONFIGURATION AND CONSTANTS
# =============================================================================

# Minimum observation threshold for sparse data handling
# PINcodes with fewer than this many total observations are flagged as sparse
MIN_OBSERVATIONS_THRESHOLD = 10

# Aggregation levels supported
AGGREGATION_LEVELS = ["year", "state", "district", "pincode"]


@dataclass
class PreprocessingResult:
    """Result container for preprocessing operations."""
    data: pd.DataFrame
    metadata: Dict
    warnings: List[str]
    
    def __str__(self) -> str:
        return (
            f"PreprocessingResult(rows={len(self.data)}, "
            f"warnings={len(self.warnings)})"
        )


# =============================================================================
# TEMPORAL AGGREGATION
# =============================================================================

def aggregate_to_yearly(
    df: pd.DataFrame,
    date_column: str = "date",
    count_columns: Optional[List[str]] = None,
    group_columns: Optional[List[str]] = None
) -> pd.DataFrame:
    """
    Aggregate daily/monthly data to yearly granularity.
    
    Assumptions:
    -----------
    1. The date column contains valid datetime values (already coerced by loaders)
    2. Count columns represent additive quantities that should be summed
    3. Geographic columns (state, district, pincode) define unique locations
    4. Aggregation preserves all geographic granularity levels
    
    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame with date column.
    date_column : str
        Name of the datetime column.
    count_columns : list, optional
        Columns to sum during aggregation. If None, inferred from column names.
    group_columns : list, optional
        Columns to group by (besides year). If None, uses geographic columns.
        
    Returns
    -------
    pd.DataFrame
        Yearly aggregated DataFrame with 'year' column replacing date.
    """
    df = df.copy()
    
    # Extract year from date
    df["year"] = df[date_column].dt.year
    
    # Infer count columns if not provided
    if count_columns is None:
        count_columns = [
            col for col in df.columns
            if col.startswith(("age_", "bio_", "demo_")) or col.endswith("_count")
        ]
    
    # Default group columns
    if group_columns is None:
        group_columns = ["state", "district", "pincode"]
    
    # Build aggregation dictionary
    agg_dict = {col: "sum" for col in count_columns if col in df.columns}
    
    # Group and aggregate
    grouped = df.groupby(["year"] + group_columns, as_index=False).agg(agg_dict)
    
    logger.info(
        f"Aggregated to yearly: {len(df)} rows -> {len(grouped)} rows "
        f"(preserved {len(group_columns)} geographic levels)"
    )
    
    return grouped


def aggregate_enrolment_yearly(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate enrolment data to yearly granularity.
    
    Creates derived columns:
    - total_enrolments: Sum of all age groups
    - child_ratio: Proportion of age 0-5 enrolments
    - minor_ratio: Proportion of age 5-17 enrolments
    - adult_ratio: Proportion of age 18+ enrolments
    
    Assumptions:
    -----------
    1. Age groups are mutually exclusive and exhaustive
    2. All counts are non-negative integers
    3. Zero total enrolments indicate no activity (not missing data)
    """
    count_cols = ["age_0_5", "age_5_17", "age_18_greater"]
    
    yearly = aggregate_to_yearly(
        df,
        date_column="date",
        count_columns=count_cols,
        group_columns=["state", "district", "pincode"]
    )
    
    # Compute total and ratios
    yearly["total_enrolments"] = (
        yearly["age_0_5"] + yearly["age_5_17"] + yearly["age_18_greater"]
    )
    
    # Safe division for ratios (avoid division by zero)
    total = yearly["total_enrolments"].replace(0, np.nan)
    yearly["child_ratio"] = yearly["age_0_5"] / total
    yearly["minor_ratio"] = yearly["age_5_17"] / total
    yearly["adult_ratio"] = yearly["age_18_greater"] / total
    
    # Fill NaN ratios with 0 (no enrolments = no distribution)
    yearly[["child_ratio", "minor_ratio", "adult_ratio"]] = (
        yearly[["child_ratio", "minor_ratio", "adult_ratio"]].fillna(0)
    )
    
    return yearly


def aggregate_biometric_updates_yearly(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate biometric update data to yearly granularity.
    
    Creates derived columns:
    - total_bio_updates: Sum of all age groups
    
    Assumptions:
    -----------
    1. bio_age_5_17 covers ages 5-17 (minors requiring biometric update)
    2. bio_age_17_ covers ages 17+ (adults/seniors)
    3. No age 0-5 category exists for biometric updates (children enrolled
       recently don't need updates, or biometrics not captured for infants)
    """
    count_cols = ["bio_age_5_17", "bio_age_17_"]
    
    yearly = aggregate_to_yearly(
        df,
        date_column="date",
        count_columns=count_cols,
        group_columns=["state", "district", "pincode"]
    )
    
    # Compute total
    yearly["total_bio_updates"] = yearly["bio_age_5_17"] + yearly["bio_age_17_"]
    
    return yearly


def aggregate_demographic_updates_yearly(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate demographic update data to yearly granularity.
    
    Creates derived columns:
    - total_demo_updates: Sum of all age groups
    
    Assumptions:
    -----------
    1. demo_age_5_17 covers ages 5-17
    2. demo_age_17_ covers ages 17+
    3. Demographic updates can occur at any age (name corrections, etc.)
    """
    count_cols = ["demo_age_5_17", "demo_age_17_"]
    
    yearly = aggregate_to_yearly(
        df,
        date_column="date",
        count_columns=count_cols,
        group_columns=["state", "district", "pincode"]
    )
    
    # Compute total
    yearly["total_demo_updates"] = yearly["demo_age_5_17"] + yearly["demo_age_17_"]
    
    return yearly


# =============================================================================
# GEOGRAPHIC CONSISTENCY
# =============================================================================

def standardize_geography(
    df: pd.DataFrame,
    state_col: str = "state",
    district_col: str = "district"
) -> Tuple[pd.DataFrame, List[str]]:
    """
    Standardize geographic names for consistency across datasets.
    
    Performs:
    - Whitespace trimming
    - Case standardization (title case)
    - Known alias resolution (e.g., "Orissa" -> "Odisha")
    
    Assumptions:
    -----------
    1. State and district names are in English
    2. Minor spelling variations represent the same location
    3. Official name changes (Orissa->Odisha) should use current names
    
    Returns
    -------
    tuple
        (standardized_df, list of transformations applied)
    """
    df = df.copy()
    warnings = []
    
    # Known state name mappings (historical -> current)
    state_aliases = {
        "ORISSA": "ODISHA",
        "PONDICHERRY": "PUDUCHERRY",
        "UTTARANCHAL": "UTTARAKHAND",
        "CHATTISGARH": "CHHATTISGARH",
    }
    
    # Standardize state names
    original_states = df[state_col].nunique()
    df[state_col] = df[state_col].str.strip().str.upper()
    df[state_col] = df[state_col].replace(state_aliases)
    new_states = df[state_col].nunique()
    
    if original_states != new_states:
        warnings.append(
            f"State standardization: {original_states} -> {new_states} unique values"
        )
    
    # Standardize district names
    original_districts = df[district_col].nunique()
    df[district_col] = df[district_col].str.strip().str.upper()
    new_districts = df[district_col].nunique()
    
    if original_districts != new_districts:
        warnings.append(
            f"District standardization: {original_districts} -> {new_districts} unique values"
        )
    
    return df, warnings


def validate_geographic_consistency(
    enrolment_df: pd.DataFrame,
    biometric_df: pd.DataFrame,
    demographic_df: pd.DataFrame
) -> Dict:
    """
    Check geographic consistency across all three datasets.
    
    Returns
    -------
    dict
        Consistency report with:
        - common_locations: Locations present in all datasets
        - enrolment_only: Locations only in enrolment data
        - updates_only: Locations only in update data (potential anomaly)
    """
    def get_locations(df):
        return set(zip(df["state"], df["district"], df["pincode"]))
    
    enrol_locs = get_locations(enrolment_df)
    bio_locs = get_locations(biometric_df)
    demo_locs = get_locations(demographic_df)
    
    all_update_locs = bio_locs | demo_locs
    
    common = enrol_locs & bio_locs & demo_locs
    enrol_only = enrol_locs - all_update_locs
    updates_only = all_update_locs - enrol_locs
    
    report = {
        "common_locations": len(common),
        "enrolment_only": len(enrol_only),
        "updates_only": len(updates_only),
        "enrolment_total": len(enrol_locs),
        "biometric_total": len(bio_locs),
        "demographic_total": len(demo_locs),
    }
    
    if updates_only:
        logger.warning(
            f"Found {len(updates_only)} locations with updates but no enrolments. "
            f"These may indicate data quality issues or cross-boundary updates."
        )
    
    return report


# =============================================================================
# SPARSE DATA HANDLING
# =============================================================================

def identify_sparse_pincodes(
    df: pd.DataFrame,
    count_column: str,
    threshold: int = MIN_OBSERVATIONS_THRESHOLD
) -> pd.DataFrame:
    """
    Identify PIN codes with sparse data below the threshold.
    
    Sparse PINcodes are those with very low observation counts, which may:
    - Have unreliable statistics
    - Represent newly created postal zones
    - Indicate data collection issues
    
    Assumptions:
    -----------
    1. Sparsity is defined by total observations, not unique observations
    2. Sparse data is flagged but NOT removed (policy decision for downstream)
    3. Threshold applies to the count column after yearly aggregation
    
    Parameters
    ----------
    df : pd.DataFrame
        Yearly aggregated DataFrame.
    count_column : str
        Column containing count to check for sparsity.
    threshold : int
        Minimum total count below which a pincode is considered sparse.
        
    Returns
    -------
    pd.DataFrame
        DataFrame with 'is_sparse' boolean column added.
    """
    df = df.copy()
    
    # Compute total counts per pincode
    pincode_totals = df.groupby("pincode")[count_column].sum()
    
    # Map sparsity flag back to rows
    df["is_sparse"] = df["pincode"].map(
        lambda x: pincode_totals.get(x, 0) < threshold
    )
    
    sparse_count = df["is_sparse"].sum()
    total_pincodes = df["pincode"].nunique()
    sparse_pincodes = df[df["is_sparse"]]["pincode"].nunique()
    
    logger.info(
        f"Sparse data identification: {sparse_pincodes}/{total_pincodes} pincodes "
        f"({sparse_count} rows) below threshold of {threshold}"
    )
    
    return df


def aggregate_sparse_to_district(
    df: pd.DataFrame,
    count_columns: List[str],
    sparse_column: str = "is_sparse"
) -> pd.DataFrame:
    """
    Aggregate sparse PIN-level data to district level.
    
    This creates a two-tier dataset:
    - Non-sparse PINcodes: Retained at PIN level
    - Sparse PINcodes: Aggregated to district level with pincode = -1
    
    Assumptions:
    -----------
    1. Sparse aggregation preserves state-district membership
    2. Pincode = -1 is a sentinel value indicating district-level aggregation
    3. This approach trades granularity for statistical reliability
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with is_sparse column.
    count_columns : list
        Columns to aggregate when combining sparse pincodes.
        
    Returns
    -------
    pd.DataFrame
        DataFrame with sparse pincodes aggregated to district level.
    """
    if sparse_column not in df.columns:
        logger.warning(f"No sparse column '{sparse_column}' found, returning unchanged")
        return df
    
    # Separate sparse and non-sparse
    non_sparse = df[~df[sparse_column]].copy()
    sparse = df[df[sparse_column]].copy()
    
    if len(sparse) == 0:
        logger.info("No sparse records to aggregate")
        return non_sparse.drop(columns=[sparse_column])
    
    # Aggregate sparse to district level
    agg_dict = {col: "sum" for col in count_columns if col in sparse.columns}
    
    sparse_agg = sparse.groupby(
        ["year", "state", "district"], as_index=False
    ).agg(agg_dict)
    
    # Mark as district-level aggregate
    sparse_agg["pincode"] = -1
    sparse_agg["aggregation_level"] = "district"
    
    # Mark non-sparse as pincode-level
    non_sparse["aggregation_level"] = "pincode"
    
    # Combine
    result = pd.concat([non_sparse.drop(columns=[sparse_column]), sparse_agg], ignore_index=True)
    
    logger.info(
        f"Sparse aggregation: {len(sparse)} sparse rows -> "
        f"{len(sparse_agg)} district-level rows"
    )
    
    return result


# =============================================================================
# NORMALIZATION
# =============================================================================

def normalize_updates_to_enrolment(
    updates_df: pd.DataFrame,
    enrolment_df: pd.DataFrame,
    update_count_col: str,
    enrolment_count_col: str = "total_enrolments"
) -> pd.DataFrame:
    """
    Normalize update counts relative to enrolled population.
    
    Creates a rate metric: updates per 1000 enrolled individuals.
    This enables fair comparison across regions with different population sizes.
    
    Assumptions:
    -----------
    1. Enrolled population in enrolment_df represents the denominator
    2. Updates can only occur for previously enrolled individuals
    3. Geographic keys (year, state, district, pincode) are consistent
    4. Division by zero handled by setting rate to NaN (flagged as no base population)
    
    Parameters
    ----------
    updates_df : pd.DataFrame
        DataFrame with update counts.
    enrolment_df : pd.DataFrame
        DataFrame with enrolment counts.
    update_count_col : str
        Column name containing update counts.
    enrolment_count_col : str
        Column name containing enrolment counts.
        
    Returns
    -------
    pd.DataFrame
        Updates DataFrame with normalized rate column added.
    """
    updates = updates_df.copy()
    
    # Prepare enrolment lookup
    join_keys = ["year", "state", "district", "pincode"]
    enrol_subset = enrolment_df[join_keys + [enrolment_count_col]].copy()
    
    # Merge enrolment data
    merged = updates.merge(enrol_subset, on=join_keys, how="left")
    
    # Calculate normalized rate (per 1000 enrolled)
    rate_col = f"{update_count_col}_rate_per_1000"
    
    # Safe division
    denominator = merged[enrolment_count_col].replace(0, np.nan)
    merged[rate_col] = (merged[update_count_col] / denominator) * 1000
    
    # Track locations with no enrolment base
    no_base = merged[rate_col].isna().sum()
    if no_base > 0:
        logger.warning(
            f"Normalization: {no_base} records have no enrolment base "
            f"(rate set to NaN)"
        )
    
    return merged


# =============================================================================
# MAIN PREPROCESSING PIPELINE
# =============================================================================

def preprocess_all_datasets(
    enrolment_df: pd.DataFrame,
    biometric_df: pd.DataFrame,
    demographic_df: pd.DataFrame,
    sparse_threshold: int = MIN_OBSERVATIONS_THRESHOLD,
    handle_sparse: str = "flag"
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict]:
    """
    Full preprocessing pipeline for all three datasets.
    
    Pipeline Steps:
    1. Standardize geographic names
    2. Aggregate to yearly granularity
    3. Identify sparse PIN-level data
    4. Optionally aggregate sparse data to district level
    5. Normalize update counts to enrolled population
    
    Parameters
    ----------
    enrolment_df : pd.DataFrame
        Raw enrolment data from loader.
    biometric_df : pd.DataFrame
        Raw biometric updates from loader.
    demographic_df : pd.DataFrame
        Raw demographic updates from loader.
    sparse_threshold : int
        Minimum observations for a pincode to be considered non-sparse.
    handle_sparse : str
        How to handle sparse data: 'flag' (add column) or 'aggregate' (to district).
        
    Returns
    -------
    tuple
        (enrolment_yearly, biometric_yearly, demographic_yearly, metadata)
    """
    all_warnings = []
    
    logger.info("Starting preprocessing pipeline...")
    
    # Step 1: Geographic standardization
    logger.info("Step 1: Geographic standardization")
    enrol, w1 = standardize_geography(enrolment_df)
    bio, w2 = standardize_geography(biometric_df)
    demo, w3 = standardize_geography(demographic_df)
    all_warnings.extend(w1 + w2 + w3)
    
    # Check geographic consistency
    geo_report = validate_geographic_consistency(enrol, bio, demo)
    
    # Step 2: Yearly aggregation
    logger.info("Step 2: Yearly aggregation")
    enrol_yearly = aggregate_enrolment_yearly(enrol)
    bio_yearly = aggregate_biometric_updates_yearly(bio)
    demo_yearly = aggregate_demographic_updates_yearly(demo)
    
    # Step 3: Sparse data handling
    logger.info("Step 3: Sparse data handling")
    enrol_yearly = identify_sparse_pincodes(
        enrol_yearly, "total_enrolments", sparse_threshold
    )
    bio_yearly = identify_sparse_pincodes(
        bio_yearly, "total_bio_updates", sparse_threshold
    )
    demo_yearly = identify_sparse_pincodes(
        demo_yearly, "total_demo_updates", sparse_threshold
    )
    
    if handle_sparse == "aggregate":
        enrol_yearly = aggregate_sparse_to_district(
            enrol_yearly,
            ["age_0_5", "age_5_17", "age_18_greater", "total_enrolments"]
        )
        bio_yearly = aggregate_sparse_to_district(
            bio_yearly,
            ["bio_age_5_17", "bio_age_17_", "total_bio_updates"]
        )
        demo_yearly = aggregate_sparse_to_district(
            demo_yearly,
            ["demo_age_5_17", "demo_age_17_", "total_demo_updates"]
        )
    
    # Step 4: Normalization
    logger.info("Step 4: Normalization")
    bio_yearly = normalize_updates_to_enrolment(
        bio_yearly, enrol_yearly, "total_bio_updates"
    )
    demo_yearly = normalize_updates_to_enrolment(
        demo_yearly, enrol_yearly, "total_demo_updates"
    )
    
    # Compile metadata
    metadata = {
        "pipeline_version": "1.0.0",
        "sparse_threshold": sparse_threshold,
        "sparse_handling": handle_sparse,
        "geographic_consistency": geo_report,
        "output_shapes": {
            "enrolment": enrol_yearly.shape,
            "biometric": bio_yearly.shape,
            "demographic": demo_yearly.shape,
        },
        "warnings": all_warnings,
    }
    
    logger.info(f"Preprocessing complete. Output shapes: {metadata['output_shapes']}")
    
    return enrol_yearly, bio_yearly, demo_yearly, metadata
