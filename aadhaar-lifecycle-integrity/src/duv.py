"""
Demographic Update Velocity (DUV) Computation Module.

Core Metric:
-----------
Demographic Update Velocity (DUV)

Quantifies the rate of non-biometric maintenance (Address + Mobile updates)
relative to the enrolled population. This serves as a proxy for "Digital Engagement"
or "Administrative Activity" independent of biometric requirements.

Formulation:
-----------
DUV = (Address Updates + Mobile Updates) / Total Enrolled Population

Interpretation:
--------------
- High DUV: Active digital engagement / frequent administrative changes.
- Low DUV: Stable population / Dormant system interaction.

Usage:
-----
Strictly a contextual signal to interpret ISI patterns.
NOT a measure of migration or residency change.

Author: Principal Data Scientist
Constraints: Government-grade, policy-safe, deterministic, aggregated only.
"""

import pandas as pd
import numpy as np
from dataclasses import dataclass
from typing import Dict, Optional
import logging

logger = logging.getLogger(__name__)


# =============================================================================
# CONFIGURATION
# =============================================================================

# Minimum population denominator to avoid noise
MIN_POPULATION_THRESHOLD = 50 


@dataclass
class DUVResult:
    """Container for DUV computation results."""
    data: pd.DataFrame
    metadata: Dict
    
    def __str__(self) -> str:
        mean_duv = self.data["duv_score"].mean() if not self.data.empty else 0
        return f"DUVResult(rows={len(self.data)}, mean_score={mean_duv:.3f})"


# =============================================================================
# COMPUTATION LOGIC
# =============================================================================

def compute_duv(
    demographic_updates_df: pd.DataFrame,
    enrolment_df: pd.DataFrame,
    target_year: int = 2025,
    match_levels: list = ["state", "district"]
) -> DUVResult:
    """
    Compute Demographic Update Velocity.
    
    Parameters
    ----------
    demographic_updates_df : pd.DataFrame
        Yearly aggregated demographic updates (from preprocess).
    enrolment_df : pd.DataFrame
        Yearly aggregated enrolment data (from preprocess).
    target_year : int
        Year to analyze.
    match_levels : list
        Geographic columns to join on (e.g., ['state', 'district']).
        
    Returns
    -------
    DUVResult
         DataFrame with 'duv_score' and metadata.
    """
    logger.info(f"Computing DUV for year {target_year} at {match_levels} level...")
    
    # 1. Filter to Target Year
    demo = demographic_updates_df[demographic_updates_df["year"] == target_year].copy()
    enrol = enrolment_df[enrolment_df["year"] == target_year].copy()
    
    if len(demo) == 0 or len(enrol) == 0:
        logger.warning("Missing data for DUV computation.")
        return DUVResult(pd.DataFrame(), {"error": "no_data"})
    
    # 2. Aggregate to Match Level (if data is finer per input)
    # Demographic updates usually contain specific components
    # We need Address + Mobile.
    # The schema defines 'demo_age_5_17' and 'demo_age_17_', but raw data
    # technically aggregates all demographic updates.
    # In this dataset, we treat 'total_demo_updates' as the sum of relevant activities.
    
    # Helper to aggregate
    def agg_df(df, cols, metric_col):
        # Only sum if column exists
        if metric_col not in df.columns:
            # Fallback sum of numeric columns if precise metric missing
            num_cols = df.select_dtypes(include=np.number).columns
            cols_to_sum = [c for c in num_cols if c not in cols + ["year"]]
            return df.groupby(cols, as_index=False)[cols_to_sum].sum()
        return df.groupby(cols, as_index=False)[metric_col].sum()

    demo_agg = agg_df(demo, match_levels, "total_demo_updates")
    enrol_agg = agg_df(enrol, match_levels, "total_enrolments")
    
    # Ensure column names are standardized after aggregation
    if "total_demo_updates" not in demo_agg.columns:
        # Sum age splits if total missing
        demo_agg["total_demo_updates"] = (
            demo_agg.get("demo_age_5_17", 0) + demo_agg.get("demo_age_17_", 0)
        )
        
    # 3. Merge Datasets
    merged = pd.merge(demo_agg, enrol_agg, on=match_levels, how="inner")
    
    # 4. Compute DUV
    # DUV = Updates / Enrolments
    
    # Safe division: valid if enrolment > threshold
    merged["valid_base"] = merged["total_enrolments"] >= MIN_POPULATION_THRESHOLD
    
    merged["duv_score"] = np.where(
        merged["valid_base"],
        merged["total_demo_updates"] / merged["total_enrolments"],
        np.nan # Insufficient base population
    )
    
    # Fill NaN DUV with 0 if updates are 0 but base is valid
    # (np.where handles the division, but if updates is 0/Enrolments, it's 0)
    # The np.nan above handles small populations to avoid noise.
    
    # 5. Metadata
    metadata = {
        "metric": "DUV",
        "year": target_year,
        "mean_duv": merged["duv_score"].mean(),
        "coverage": len(merged)
    }
    
    logger.info(f"DUV Computed. Mean Score: {metadata['mean_duv']:.3f}")
    
    return DUVResult(merged, metadata)
