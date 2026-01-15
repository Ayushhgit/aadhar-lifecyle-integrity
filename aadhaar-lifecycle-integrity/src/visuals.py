"""
Visualization utilities for Aadhaar lifecycle analysis.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Optional, Tuple
import logging

from . import config

logger = logging.getLogger(__name__)


def setup_style():
    """Configure matplotlib style for consistent visualizations."""
    plt.style.use("seaborn-v0_8-whitegrid")
    plt.rcParams.update({
        "figure.figsize": (10, 6),
        "figure.dpi": config.FIGURE_DPI,
        "font.size": 12,
        "axes.titlesize": 14,
        "axes.labelsize": 12,
    })


def save_figure(fig, filename: str, output_dir: Optional[Path] = None):
    """Save figure to output directory."""
    if output_dir is None:
        output_dir = config.FIGURES_DIR
    output_dir.mkdir(parents=True, exist_ok=True)
    filepath = output_dir / f"{filename}.{config.FIGURE_FORMAT}"
    fig.savefig(filepath, bbox_inches="tight", dpi=config.FIGURE_DPI)
    logger.info(f"Saved figure to {filepath}")
    plt.close(fig)


def plot_isi_distribution(df: pd.DataFrame, isi_column: str = "isi", save: bool = True):
    """Plot ISI score distribution."""
    setup_style()
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    sns.histplot(df[isi_column], kde=True, ax=axes[0], color="steelblue")
    axes[0].set_xlabel("ISI Score")
    axes[0].set_title("ISI Distribution")
    
    sns.boxplot(x=df[isi_column], ax=axes[1], color="steelblue")
    axes[1].set_xlabel("ISI Score")
    axes[1].set_title("ISI Box Plot")
    
    plt.tight_layout()
    if save:
        save_figure(fig, "isi_distribution")
    return fig


def plot_duv_distribution(df: pd.DataFrame, duv_column: str = "duv", save: bool = True):
    """Plot DUV distribution."""
    setup_style()
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.histplot(df[duv_column], kde=True, ax=ax, color="coral")
    ax.set_xlabel("Data Update Velocity")
    ax.set_title("DUV Distribution")
    if save:
        save_figure(fig, "duv_distribution")
    return fig


def plot_isi_duv_scatter(df: pd.DataFrame, save: bool = True):
    """Plot ISI vs DUV scatter plot."""
    setup_style()
    fig, ax = plt.subplots(figsize=(10, 8))
    scatter = ax.scatter(df["isi"], df["duv"], c=df.get("risk_score", None), 
                        cmap="RdYlGn_r", alpha=0.6, s=50)
    ax.set_xlabel("Integrity Score Index (ISI)")
    ax.set_ylabel("Data Update Velocity (DUV)")
    ax.set_title("ISI vs DUV Analysis")
    if "risk_score" in df.columns:
        plt.colorbar(scatter, label="Risk Score")
    if save:
        save_figure(fig, "isi_duv_scatter")
    return fig


def plot_cohort_trends(df: pd.DataFrame, cohort_col: str, value_col: str, save: bool = True):
    """Plot trends across cohorts."""
    setup_style()
    fig, ax = plt.subplots(figsize=(12, 6))
    cohort_means = df.groupby(cohort_col)[value_col].mean()
    cohort_means.plot(kind="line", marker="o", ax=ax, color="teal")
    ax.set_xlabel("Cohort")
    ax.set_ylabel(value_col)
    ax.set_title(f"{value_col} Trend by Cohort")
    plt.xticks(rotation=45)
    if save:
        save_figure(fig, f"cohort_trend_{value_col}")
    return fig


def plot_risk_heatmap(df: pd.DataFrame, save: bool = True):
    """Plot risk heatmap by category."""
    setup_style()
    fig, ax = plt.subplots(figsize=(10, 8))
    if "isi_category" in df.columns and "duv_category" in df.columns:
        pivot = pd.crosstab(df["isi_category"], df["duv_category"])
        sns.heatmap(pivot, annot=True, fmt="d", cmap="YlOrRd", ax=ax)
        ax.set_title("Risk Matrix: ISI vs DUV Categories")
    if save:
        save_figure(fig, "risk_heatmap")
    return fig


def plot_geographic_distribution(df: pd.DataFrame, state_col: str = "state_code", save: bool = True):
    """Plot geographic distribution of records."""
    setup_style()
    fig, ax = plt.subplots(figsize=(14, 6))
    state_counts = df[state_col].value_counts().head(20)
    state_counts.plot(kind="bar", ax=ax, color="steelblue")
    ax.set_xlabel("State Code")
    ax.set_ylabel("Count")
    ax.set_title("Top 20 States by Record Count")
    plt.xticks(rotation=45)
    if save:
        save_figure(fig, "geographic_distribution")
    return fig


def save_table(df: pd.DataFrame, filename: str, output_dir: Optional[Path] = None):
    """Save DataFrame as CSV to tables directory."""
    if output_dir is None:
        output_dir = config.TABLES_DIR
    output_dir.mkdir(parents=True, exist_ok=True)
    filepath = output_dir / f"{filename}.csv"
    df.to_csv(filepath, index=False)
    logger.info(f"Saved table to {filepath}")
