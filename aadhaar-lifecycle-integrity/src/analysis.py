"""
Joint Diagnostic Analysis Module (ISI x DUV).

Purpose:
-------
Integrates Cross-Sectional Identity Staleness Index (ISI) and
Demographic Update Velocity (DUV) to produce a unified diagnostic.

Quadrant Classification:
-----------------------
Classifies administrative units into 4 System States:

1. [High ISI + Low DUV] -> "Infrastructure Gap / Access Constraint"
   (Maintenance failing, general engagement low)

2. [High ISI + High DUV] -> "High-Engagement, High-Friction Zone"
   (Population active digitally, but biometrics lagging)

3. [Low ISI + Low DUV] -> "Structurally Stable / Dormant"
   (Everything quiet, balanced)

4. [Low ISI + High DUV] -> "Digitally Engaged & Balanced"
   (Active population, biometrics keeping pace)

Author: Principal Data Scientist
Constraints: Government-grade, policy-safe, district-level execution.
"""

import pandas as pd
import numpy as np
from typing import Dict, Tuple, Optional
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


# =============================================================================
# CONFIGURATION
# =============================================================================

# Thresholds for High/Low Classification
ISI_HIGH_THRESHOLD = 0.4  # Matches Risk Class HIGH/CRITICAL bound
DUV_HIGH_THRESHOLD = 0.05 # 5% annual update rate proxy (configurable)


@dataclass
class JointDiagnosticResult:
    """Container for joint analysis results."""
    data: pd.DataFrame
    metadata: Dict
    
    def __str__(self) -> str:
        return f"JointDiagnosticResult(rows={len(self.data)})"


# =============================================================================
# JOINT ANALYSIS LOGIC
# =============================================================================

def classify_quadrant(row, isi_thresh=ISI_HIGH_THRESHOLD, duv_thresh=DUV_HIGH_THRESHOLD):
    """Assign diagnostic quadrant based on ISI and DUV scores."""
    isi = row.get("isi_bounded", 0)
    duv = row.get("duv_score", 0)
    
    # Check nulls
    if pd.isna(isi) or pd.isna(duv):
        return "Insufficient Data"
        
    is_isi_high = isi > isi_thresh
    is_duv_high = duv > duv_thresh
    
    if is_isi_high and not is_duv_high:
        return "Infrastructure Gap / Access Constraint"
    elif is_isi_high and is_duv_high:
        return "High-Engagement, High-Friction Zone"
    elif not is_isi_high and not is_duv_high:
        return "Structurally Stable / Dormant System"
    else: # Low ISI, High DUV
        return "Digitally Engaged & Balanced System"


def perform_joint_analysis(
    isi_df: pd.DataFrame,
    duv_df: pd.DataFrame,
    merge_keys: list = ["state", "district"]
) -> JointDiagnosticResult:
    """
    Merge ISI and DUV datasets and perform quadrant classification.
    
    Parameters
    ----------
    isi_df : pd.DataFrame
        DataFrame containing 'isi_bounded', 'isi_raw', 'risk_level'.
    duv_df : pd.DataFrame
        DataFrame containing 'duv_score'.
    merge_keys : list
        Columns to join on.
        
    Returns
    -------
    JointDiagnosticResult
        Merged DataFrame with 'diagnostic_quadrant' column.
    """
    logger.info("Starting Joint ISI x DUV Analysis...")
    
    # 1. Merge
    # Use inner join to ensure we only analyze units with BOTH signals
    merged = pd.merge(
        isi_df, 
        duv_df[merge_keys + ["duv_score"]], 
        on=merge_keys, 
        how="inner"
    )
    
    if len(merged) == 0:
        logger.error("Merge failed. Check geographic keys.")
        return JointDiagnosticResult(pd.DataFrame(), {"error": "merge_empty"})
    
    # 2. Quadrant Classification
    merged["diagnostic_quadrant"] = merged.apply(classify_quadrant, axis=1)
    
    # 3. Summary Statistics
    quadrant_counts = merged["diagnostic_quadrant"].value_counts().to_dict()
    
    metadata = {
        "isi_threshold": ISI_HIGH_THRESHOLD,
        "duv_threshold": DUV_HIGH_THRESHOLD,
        "quadrant_distribution": quadrant_counts,
        "mean_isi": merged["isi_bounded"].mean(),
        "mean_duv": merged["duv_score"].mean()
    }
    
    logger.info("Joint Analysis Complete.")
    logger.info(f"Quadrant Distribution: {quadrant_counts}")
    
    return JointDiagnosticResult(merged, metadata)
