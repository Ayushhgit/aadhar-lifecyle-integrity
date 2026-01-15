"""
Statistical analysis and correlation utilities.
"""

import pandas as pd
import numpy as np
from scipy import stats
from typing import Dict, List, Optional
import logging

logger = logging.getLogger(__name__)


def compute_correlation_matrix(
    df: pd.DataFrame,
    columns: Optional[List[str]] = None,
    method: str = "pearson"
) -> pd.DataFrame:
    """Compute correlation matrix for numerical columns."""
    if columns is None:
        columns = df.select_dtypes(include=[np.number]).columns.tolist()
    return df[columns].corr(method=method)


def compute_isi_duv_correlation(
    df: pd.DataFrame,
    isi_column: str = "isi",
    duv_column: str = "duv"
) -> Dict:
    """Compute correlation between ISI and DUV."""
    valid_data = df[[isi_column, duv_column]].dropna()
    pearson_r, pearson_p = stats.pearsonr(valid_data[isi_column], valid_data[duv_column])
    spearman_r, spearman_p = stats.spearmanr(valid_data[isi_column], valid_data[duv_column])
    return {
        "pearson_correlation": pearson_r,
        "pearson_pvalue": pearson_p,
        "spearman_correlation": spearman_r,
        "spearman_pvalue": spearman_p,
        "n_samples": len(valid_data)
    }


def compute_group_statistics(
    df: pd.DataFrame,
    group_column: str,
    value_columns: List[str]
) -> pd.DataFrame:
    """Compute descriptive statistics by group."""
    return df.groupby(group_column)[value_columns].agg(["count", "mean", "std", "min", "median", "max"])


def perform_hypothesis_test(group1: pd.Series, group2: pd.Series, test: str = "ttest") -> Dict:
    """Perform hypothesis test between two groups."""
    group1, group2 = group1.dropna(), group2.dropna()
    if test == "ttest":
        stat, pval = stats.ttest_ind(group1, group2)
    elif test == "mannwhitney":
        stat, pval = stats.mannwhitneyu(group1, group2)
    else:
        stat, pval = stats.ks_2samp(group1, group2)
    return {"statistic": stat, "pvalue": pval}


def compute_joint_analysis(isi_df: pd.DataFrame, duv_df: pd.DataFrame, join_column: str = "aadhaar_number") -> pd.DataFrame:
    """Perform joint ISI-DUV analysis."""
    joint = isi_df.merge(duv_df, on=join_column, how="inner")
    isi_med, duv_med = joint["isi"].median(), joint.get("duv", pd.Series([0.5])).median()
    conditions = [
        (joint["isi"] >= isi_med) & (joint["duv"] >= duv_med),
        (joint["isi"] >= isi_med) & (joint["duv"] < duv_med),
        (joint["isi"] < isi_med) & (joint["duv"] >= duv_med),
    ]
    joint["quadrant"] = np.select(conditions, ["HI_HI", "HI_LO", "LO_HI"], default="LO_LO")
    return joint
