"""
Risk Classification and Policy Action Module.

This module maps quantitative Identity Staleness Index (ISI) values to
actionable, policy-safe administrative interventions.

Core Principle:
--------------
Converts numerical diagnostics into system-level operational signals.
Categorizations represent data freshness states, NOT resident behavior.

Risk Bands:
----------
- CRITICAL (> 0.7 gap): High staleness, urgent system intervention needed
- HIGH (0.5 - 0.7 gap): Significant lag, targeted camps required
- MODERATE (0.2 - 0.5 gap): Emerging staleness, routine maintenance needed
- LOW (< 0.2 gap): Healthy system state, maintain current operations

Author: Principal Data Scientist
Constraints: Government-grade, policy-safe, deterministic
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional
from dataclasses import dataclass
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class RiskLevel(Enum):
    """System-level risk categories based on staleness gap."""
    CRITICAL = "CRITICAL"
    HIGH = "HIGH"
    MODERATE = "MODERATE"
    LOW = "LOW"
    UNKNOWN = "UNKNOWN"


@dataclass
class RiskBandDef:
    """Definition of a risk band with associated actions."""
    level: RiskLevel
    min_gap_ratio: float
    max_gap_ratio: float
    description: str
    admin_actions: List[str]


# =============================================================================
# RISK BAND DEFINITIONS
# =============================================================================

RISK_BANDS = [
    RiskBandDef(
        level=RiskLevel.CRITICAL,
        min_gap_ratio=0.7,
        max_gap_ratio=float('inf'),
        description="Severe biometric staleness detected. Majority of cohort lacks expected updates.",
        admin_actions=[
            "Deploy mobile biometric update units to high-density clusters",
            "Schedule mandatory school-based biometric camps (ages 5-15)",
            "Initiate district-level review of enrolment centre availability"
        ]
    ),
    RiskBandDef(
        level=RiskLevel.HIGH,
        min_gap_ratio=0.5,
        max_gap_ratio=0.7,
        description="Significant maintenance lag. Update velocity well below cohort accumulation.",
        admin_actions=[
            "Prioritize face authentication rollout to mitigate fingerprint staleness",
            "Increase frequency of Anganwadi/School update drives",
            "Send localized system-level notifications to local registrars"
        ]
    ),
    RiskBandDef(
        level=RiskLevel.MODERATE,
        min_gap_ratio=0.2,
        max_gap_ratio=0.5,
        description="Emerging staleness gap. Routine operations insufficient for backlog.",
        admin_actions=[
            "Optimize existing enrolment centre operating hours",
            "Conduct targeted awareness campaigns on update benefits",
            "Monitor rejection rates for potential equipment issues"
        ]
    ),
    RiskBandDef(
        level=RiskLevel.LOW,
        min_gap_ratio=-float('inf'), # Negative gap means more updates than expected (good)
        max_gap_ratio=0.2,
        description="System operating within healthy maintenance parameters.",
        admin_actions=[
            "Maintain current operational cadence",
            "Conduct random quality audits to ensure update validity",
            "Analyze best practices for replication in other districts"
        ]
    )
]


# =============================================================================
# CLASSIFICATION FUNCTIONS
# =============================================================================

def get_risk_band(gap_ratio: float) -> RiskBandDef:
    """
    Map a gap ratio to the appropriate risk band definition.
    
    Parameters
    ----------
    gap_ratio : float
        The ratio of missing updates to expected updates (1 - update_ratio).
        Values close to 1.0 indicate high staleness (missing almost all updates).
        Values <= 0 indicate healthy state (observed >= expected).
        
    Returns
    -------
    RiskBandDef
        The matching risk band definition.
    """
    if pd.isna(gap_ratio):
        return RiskBandDef(
            level=RiskLevel.UNKNOWN,
            min_gap_ratio=0, max_gap_ratio=0,
            description="Insufficient data to calculate risk.",
            admin_actions=["Verify data completeness"]
        )
        
    for band in RISK_BANDS:
        if band.min_gap_ratio <= gap_ratio < band.max_gap_ratio:
            return band
            
    # Fallback (should cover all ranges given -inf/+inf)
    return RISK_BANDS[-1]


def classify_staleness_risk(
    cohort_df: pd.DataFrame,
    gap_column: str = "update_gap_ratio"
) -> pd.DataFrame:
    """
    Apply risk classification to cohort alignment data.
    
    Assumptions:
    -----------
    1. Input DataFrame contains a gap ratio column describing staleness
       (where 1.0 = 100% stale, 0.0 = 0% stale/fresh)
    2. If update_ratio is provided instead of gap_ratio, gap is calculated
       as (1 - update_ratio) clipping negative values to 0 (over-performance).
    
    Parameters
    ----------
    cohort_df : pd.DataFrame
        Input DataFrame with cohort alignment data.
    gap_column : str
        Name of the column to store/read the staleness gap ratio.
        
    Returns
    -------
    pd.DataFrame
        DataFrame with added risk diagnostic columns:
        - risk_level: CRITICAL, HIGH, MODERATE, LOW
        - risk_description: Plain language system diagnostic
        - recommended_action_primary: Policy-safe administrative action
    """
    df = cohort_df.copy()
    
    # Calculate gap ratio if implied by update_ratio
    # update_ratio = observed / expected
    # gap_ratio = (expected - observed) / expected = 1 - update_ratio
    # High gap ratio (>0) is bad (under-maintenance)
    if "update_ratio" in df.columns and gap_column not in df.columns:
        # Cap update ratio at 1.0 for risk purposes (over-performance is 0 risk)
        # Handle division by zero/NaN implicitly handled by numpy
        df[gap_column] = 1.0 - df["update_ratio"].fillna(0)
        # Negative gap ratio means we have MORE updates than expected (good)
        # We allow negative values to flow to LOW risk band (-inf to 0.2)
    
    # Define vectorizable mapping functions
    def map_level(ratio):
        return get_risk_band(ratio).level.value
        
    def map_desc(ratio):
        return get_risk_band(ratio).description
        
    def map_action(ratio):
        return get_risk_band(ratio).admin_actions[0]  # Return primary action
    
    # Apply mapping
    df["risk_level"] = df[gap_column].apply(map_level)
    df["risk_description"] = df[gap_column].apply(map_desc)
    df["recommended_action_primary"] = df[gap_column].apply(map_action)
    
    # Log summary
    risk_counts = df["risk_level"].value_counts()
    logger.info(f"Risk classification complete. Summary:\n{risk_counts}")
    
    return df


def generate_diagnostic_report(classified_df: pd.DataFrame) -> Dict:
    """
    Generate a system-level diagnostic report from classified data.
    
    Returns high-level statistics suitable for administrative dashboards.
    
    Parameters
    ----------
    classified_df : pd.DataFrame
        Dataframe with risk classifications.
        
    Returns
    -------
    dict
        Diagnostic summary dictionary.
    """
    total_locations = len(classified_df)
    critical_locs = len(classified_df[classified_df["risk_level"] == "CRITICAL"])
    
    report = {
        "system_health_index": 1.0 - (critical_locs / total_locations if total_locations > 0 else 0),
        "critical_districts_count": critical_locs,
        "risk_distribution": classified_df["risk_level"].value_counts().to_dict(),
        "primary_intervention_needed": classified_df[
            classified_df["risk_level"] == "CRITICAL"
        ]["recommended_action_primary"].mode().iloc[0] if critical_locs > 0 else "None",
    }
    
    return report
