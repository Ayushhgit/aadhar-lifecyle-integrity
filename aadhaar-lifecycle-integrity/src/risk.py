"""
Risk assessment and scoring module.
"""

import pandas as pd
import numpy as np
from typing import Dict, Optional, List
import logging

logger = logging.getLogger(__name__)


def compute_risk_score(
    isi_score: float,
    duv_score: float,
    isi_weight: float = 0.6,
    duv_weight: float = 0.4
) -> float:
    """
    Compute combined risk score from ISI and DUV.
    
    Lower ISI and abnormal DUV indicate higher risk.
    
    Parameters
    ----------
    isi_score : float
        Integrity Score Index (0-1).
    duv_score : float
        Data Update Velocity score (0-1).
    isi_weight : float
        Weight for ISI component.
    duv_weight : float
        Weight for DUV component.
        
    Returns
    -------
    float
        Risk score (0-1, higher is riskier).
    """
    # Invert ISI and DUV scores to get risk
    isi_risk = 1 - isi_score
    duv_risk = 1 - duv_score
    
    return isi_weight * isi_risk + duv_weight * duv_risk


def compute_risk_matrix(
    df: pd.DataFrame,
    isi_column: str = "isi",
    duv_column: str = "duv_score"
) -> pd.DataFrame:
    """
    Compute risk matrix for all records.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with ISI and DUV scores.
    isi_column : str
        Name of ISI column.
    duv_column : str
        Name of DUV column.
        
    Returns
    -------
    pd.DataFrame
        DataFrame with risk scores and categories.
    """
    df = df.copy()
    
    # Compute individual risk scores
    df["risk_score"] = df.apply(
        lambda row: compute_risk_score(
            row.get(isi_column, 0.5),
            row.get(duv_column, 0.5)
        ),
        axis=1
    )
    
    # Categorize risk
    df["risk_category"] = pd.cut(
        df["risk_score"],
        bins=[0, 0.25, 0.5, 0.75, 1.0],
        labels=["LOW", "MEDIUM", "HIGH", "CRITICAL"],
        include_lowest=True
    )
    
    return df


def identify_anomalies(
    df: pd.DataFrame,
    columns: List[str],
    method: str = "zscore",
    threshold: float = 3.0
) -> pd.DataFrame:
    """
    Identify anomalous records based on statistical methods.
    
    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame.
    columns : list
        Columns to check for anomalies.
    method : str
        Detection method: 'zscore' or 'iqr'.
    threshold : float
        Threshold for anomaly detection.
        
    Returns
    -------
    pd.DataFrame
        DataFrame with anomaly flag.
    """
    df = df.copy()
    df["is_anomaly"] = False
    
    for col in columns:
        if col not in df.columns:
            continue
            
        if method == "zscore":
            z_scores = np.abs((df[col] - df[col].mean()) / df[col].std())
            df.loc[z_scores > threshold, "is_anomaly"] = True
        elif method == "iqr":
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower = Q1 - threshold * IQR
            upper = Q3 + threshold * IQR
            df.loc[(df[col] < lower) | (df[col] > upper), "is_anomaly"] = True
    
    anomaly_count = df["is_anomaly"].sum()
    logger.info(f"Identified {anomaly_count} anomalies ({anomaly_count/len(df)*100:.2f}%)")
    
    return df


def compute_geographic_risk(
    df: pd.DataFrame,
    state_column: str = "state_code",
    district_column: str = "district_code"
) -> pd.DataFrame:
    """
    Compute geographic risk concentration.
    
    Parameters
    ----------
    df : pd.DataFrame
        Data with geographic codes.
    state_column : str
        Name of state column.
    district_column : str
        Name of district column.
        
    Returns
    -------
    pd.DataFrame
        Geographic risk metrics.
    """
    if state_column not in df.columns:
        return pd.DataFrame()
    
    # Calculate risk by geography
    geo_risk = df.groupby([state_column, district_column]).agg({
        "risk_score": ["mean", "count"],
        "is_anomaly": "sum"
    })
    geo_risk.columns = ["mean_risk", "record_count", "anomaly_count"]
    geo_risk["anomaly_rate"] = geo_risk["anomaly_count"] / geo_risk["record_count"]
    
    return geo_risk.reset_index()


def compute_temporal_risk(
    df: pd.DataFrame,
    date_column: str,
    frequency: str = "M"
) -> pd.DataFrame:
    """
    Compute temporal risk trends.
    
    Parameters
    ----------
    df : pd.DataFrame
        Data with date information.
    date_column : str
        Name of the date column.
    frequency : str
        Aggregation frequency.
        
    Returns
    -------
    pd.DataFrame
        Temporal risk metrics.
    """
    if date_column not in df.columns:
        return pd.DataFrame()
    
    df = df.copy()
    df["period"] = df[date_column].dt.to_period(frequency)
    
    temporal_risk = df.groupby("period").agg({
        "risk_score": ["mean", "std"],
        "is_anomaly": ["sum", "mean"]
    })
    temporal_risk.columns = ["mean_risk", "std_risk", "anomaly_count", "anomaly_rate"]
    
    return temporal_risk.reset_index()


def generate_risk_report(
    df: pd.DataFrame,
    top_n: int = 10
) -> Dict:
    """
    Generate a summary risk report.
    
    Parameters
    ----------
    df : pd.DataFrame
        Data with risk scores.
    top_n : int
        Number of top risky records to highlight.
        
    Returns
    -------
    dict
        Risk report summary.
    """
    report = {
        "total_records": len(df),
        "mean_risk_score": df["risk_score"].mean(),
        "std_risk_score": df["risk_score"].std(),
        "risk_distribution": df["risk_category"].value_counts().to_dict(),
        "anomaly_count": df["is_anomaly"].sum() if "is_anomaly" in df.columns else 0,
        "top_risky_records": df.nlargest(top_n, "risk_score")[
            ["aadhaar_number", "risk_score", "risk_category"]
        ].to_dict("records")
    }
    
    return report
