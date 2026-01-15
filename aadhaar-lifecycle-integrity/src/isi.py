"""
Identity Staleness Index (ISI) Computation Module.

Core Metric:
-----------
Cross-Sectional Identity Staleness Index (ISI).

Quantifies the structural lag between "Entry Pressure" (0-5 Enrolments) and
"Maintenance Volume" (5-17 Biometric Updates) within the same administrative year.

Formulation:
-----------
ISI_loc = 1 - ( (Observed + epsilon) / (Expected + epsilon) )

Reporting Bounds:
----------------
ISI* = max(0, min(ISI, 1))

Interpretation:
--------------
ISI* = 0  : Structurally Balanced (Healthy)
ISI* > 0  : Identity Staleness (Lag)
Raw < 0   : Catch-up Zone (Backlog Clearance)

Author: Principal Data Scientist
Constraints: Government-grade, policy-safe, deterministic
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

# Epsilon for numerical stability
# Small enough to not skew large counts, large enough to handle zeros
EPSILON = 1e-6


@dataclass
class ISIResult:
    """Container for ISI computation results."""
    data: pd.DataFrame
    metadata: Dict
    
    def __str__(self) -> str:
        mean_isi = self.data["isi_bounded"].mean() if not self.data.empty else 0
        return f"ISIResult(rows={len(self.data)}, mean_score={mean_isi:.3f})"


# =============================================================================
# COMPUTATION LOGIC
# =============================================================================

def compute_isi(
    cohort_df: pd.DataFrame,
    epsilon: float = EPSILON
) -> ISIResult:
    """
    Compute Identity Staleness Index from aligned cohort flows.
    
    Parameters
    ----------
    cohort_df : pd.DataFrame
        Output from cohort_alignment.perform_cohort_alignment.
        Must contain 'expected_updates' and 'observed_updates'.
    epsilon : float
        Stability constant for division.
        
    Returns
    -------
    ISIResult
        DataFrame with 'isi_raw', 'isi_bounded', and 'interpretation'.
    """
    df = cohort_df.copy()
    
    # Validation
    required_cols = ["expected_updates", "observed_updates"]
    if not all(col in df.columns for col in required_cols):
        logger.error(f"Missing required columns: {required_cols}")
        return ISIResult(pd.DataFrame(), {"error": "missing_columns"})
    
    # 1. Compute Flow Ratio (Consolidated Throughput)
    # Ratio = (Observed + e) / (Expected + e)
    df["flow_ratio"] = (
        (df["observed_updates"] + epsilon) / 
        (df["expected_updates"] + epsilon)
    )
    
    # 2. Compute Raw ISI
    # ISI = 1 - Ratio
    # If Ratio = 1.0 (Balanced), ISI = 0.0
    # If Ratio = 0.5 (Lag), ISI = 0.5
    # If Ratio = 2.0 (Catch-up), ISI = -1.0
    df["isi_raw"] = 1.0 - df["flow_ratio"]
    
    # 3. Compute Bounded ISI (Reporting Metric)
    # Clip to [0, 1]
    df["isi_bounded"] = df["isi_raw"].clip(lower=0.0, upper=1.0)
    
    # 4. Diagnostic Categories
    conditions = [
        (df["isi_raw"] < 0),                    # Catch-up (More updates than expected)
        (df["isi_bounded"] <= 0.1),             # Balanced (allow small noise margin)
        (df["isi_bounded"] > 0.1) & (df["isi_bounded"] <= 0.4), # Moderate Staleness
        (df["isi_bounded"] > 0.4)               # High Staleness
    ]
    choices = [
        "Catch-up Zone",
        "Structurally Balanced",
        "Moderate Staleness",
        "High Staleness"
    ]
    
    df["isi_category"] = np.select(conditions, choices, default="Unknown")
    
    # Metadata
    metadata = {
        "metric": "Cross-Sectional ISI",
        "epsilon": epsilon,
        "mean_isi": df["isi_bounded"].mean(),
        "catch_up_zones": int((df["isi_raw"] < 0).sum())
    }
    
    logger.info(
        f"ISI Computation Complete. Mean Score: {metadata['mean_isi']:.3f}. "
        f"Catch-up Zones detected: {metadata['catch_up_zones']}"
    )
    
    return ISIResult(df, metadata)
