"""
Risk Classification and Policy Action Module.

This module maps Cross-Sectional Identity Staleness Index (ISI) values to
actionable, policy-safe administrative interventions.

Core formulation:
ISI = 1 - (Observed / Expected)

Risk Bands:
----------
- CRITICAL (ISI > 0.7): Severe maintenance failure
- HIGH (0.4 < ISI <= 0.7): Significant maintenance lag
- MODERATE (0.1 < ISI <= 0.4): Emerging lag
- BALANCED (0.0 <= ISI <= 0.1): Healthy structural flow
- CATCH-UP (ISI < 0): Backlog clearance activity

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
    """System-level risk categories."""
    CRITICAL = "CRITICAL"
    HIGH = "HIGH"
    MODERATE = "MODERATE"
    BALANCED = "BALANCED"
    CATCH_UP = "CATCH-UP" # Special category for negative ISI
    UNKNOWN = "UNKNOWN"


@dataclass
class RiskBandDef:
    """Definition of a risk band with associated actions."""
    level: RiskLevel
    min_isi: float
    max_isi: float
    description: str
    admin_actions: List[str]


# =============================================================================
# RISK BAND DEFINITIONS
# =============================================================================

RISK_BANDS = [
    RiskBandDef(
        level=RiskLevel.CRITICAL,
        min_isi=0.7,
        max_isi=float('inf'),
        description="Severe structural lag. Maintenance volume critically below entry pressure.",
        admin_actions=[
            "Deploy mobile biometric update units to high-entry districts",
            "Mandate school-based biometric camps (ages 5-15)",
            "Review registrar capacity for biometric maintenance"
        ]
    ),
    RiskBandDef(
        level=RiskLevel.HIGH,
        min_isi=0.4,
        max_isi=0.7,
        description="Significant maintenance lag. Throughput failing to match cohort entry.",
        admin_actions=[
            "Prioritize face authentication rollout",
            "Increase frequency of Anganwadi update drives",
            "Issue system-level throughput notifications"
        ]
    ),
    RiskBandDef(
        level=RiskLevel.MODERATE,
        min_isi=0.1,
        max_isi=0.4,
        description="Emerging consistency gap. Throughput slightly trailing entry.",
        admin_actions=[
            "Optimize enrolment centre operating hours",
            "Conduct awareness campaigns on update benefits",
            "Monitor rejection rates"
        ]
    ),
    RiskBandDef(
        level=RiskLevel.BALANCED,
        min_isi=0.0,
        max_isi=0.1,
        description="Structurally balanced system. Maintenance usage scales with entry.",
        admin_actions=[
            "Maintain current operational cadence",
            "Conduct quality audits",
            "Document best practices"
        ]
    ),
    RiskBandDef(
        level=RiskLevel.CATCH_UP,
        min_isi=-float('inf'),
        max_isi=0.0,
        description="Catch-up Zone. Maintenance volume exceeds entry pressure (backlog clearance).",
        admin_actions=[
            "Monitor operator load for burnout risks",
            "Ensure quality assurance on high-volume updates",
            "Analyze factors driving high compliance"
        ]
    )
]


# =============================================================================
# CLASSIFICATION FUNCTIONS
# =============================================================================

def get_risk_band(isi_value: float) -> RiskBandDef:
    """Map an ISI value to a risk band."""
    if pd.isna(isi_value):
        return RiskBandDef(
            level=RiskLevel.UNKNOWN, min_isi=0, max_isi=0,
            description="Insufficient data", admin_actions=["Verify data"]
        )
        
    for band in RISK_BANDS:
        # Use raw isi logic (including negative)
        if band.min_isi <= isi_value < band.max_isi:
            return band
            
    # Default fallback (should catch edge cases like exact upper bounds if not caught)
    # Logic: if extremely high, critical. if extremely low, catch up.
    if isi_value >= 0.7: return RISK_BANDS[0] # Critical
    if isi_value < 0: return RISK_BANDS[4]    # Catch Up
    
    return RISK_BANDS[3] # Default to Balanced if falls through cracks


def classify_staleness_risk(
    isi_df: pd.DataFrame,
    isi_column: str = "isi_raw"
) -> pd.DataFrame:
    """
    Apply risk classification to ISI Dataframe.
    
    Parameters
    ----------
    isi_df : pd.DataFrame
        DataFrame with ISI scores (e.g. from isi.py).
    isi_column : str
        Column to use for classification. Recommended: 'isi_raw' to capture
        catch-up behavior (negative values).
        
    Returns
    -------
    pd.DataFrame
        Dataframe with risk columns added.
    """
    df = isi_df.copy()
    
    if isi_column not in df.columns:
        logger.warning(f"ISI column {isi_column} not found via risk classification.")
        return df
    
    # Vectorized mapping using helper
    # (Using apply is simpler for readable code here than np.select for complex objects)
    
    def map_row(val):
        band = get_risk_band(val)
        return pd.Series([
            band.level.value,
            band.description,
            band.admin_actions[0]
        ])
        
    df[["risk_level", "risk_description", "recommended_action_primary"]] = (
        df[isi_column].apply(map_row)
    )
    
    # Counts
    logger.info("Risk Classification Summary:")
    logger.info(df["risk_level"].value_counts())
    
    return df


def generate_diagnostic_report(classified_df: pd.DataFrame) -> Dict:
    """Generate high-level system diagnostic statistics."""
    total = len(classified_df)
    critical = len(classified_df[classified_df["risk_level"] == "CRITICAL"])
    catch_up = len(classified_df[classified_df["risk_level"] == "CATCH-UP"])
    
    # Health Index: Proportion NOT Critical
    health = 1.0 - (critical / total) if total > 0 else 0
    
    report = {
        "system_health_index": health,
        "critical_units": critical,
        "catch_up_units": catch_up,
        "risk_distribution": classified_df["risk_level"].value_counts().to_dict()
    }
    
    return report
