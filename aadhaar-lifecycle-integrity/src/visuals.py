"""
Visualization and Reporting Module (Decision-Grade).

Purpose:
-------
Generates high-impact, policy-safe visualizations for Aadhaar Lifecycle Analysis.
Focuses on visual salience of critical districts while preserving data integrity.

Outputs:
-------
1. Figures (PNG):
   - ISI Distribution (with Critical Tail Inset)
   - ISI x DUV Quadrant Diagnostic (Annotated)
   - Top-10 Intervention Candidates (Bar Chart)

Design Constraints:
------------------
- Colorblind-safe, print-friendly palettes
- Explicit threshold shading
- Decision-centric titles and annotations

Author: Principal Data Visualization Engineer
"""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from pathlib import Path
import logging
from typing import Dict, List, Optional
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from src.analysis import ISI_HIGH_THRESHOLD, DUV_HIGH_THRESHOLD

logger = logging.getLogger(__name__)

# =============================================================================
# CONFIGURATION
# =============================================================================

# Style Settings
plt.style.use("seaborn-v0_8-white") # Cleaner, less grid-heavy
sns.set_context("talk", font_scale=1.0)

# Decision-Grade Palette
# Healthy = Muted Blue/Grey
# Critical = Saturated Red/Orange
QUADRANT_COLORS = {
    "Digitally Engaged & Balanced System": "#95a5a6",      # Muted Grey-Blue (Safe)
    "Structurally Stable / Dormant System": "#7f8c8d",     # Muted Dark Grey
    "High-Engagement, High-Friction Zone": "#d35400",      # Burnt Orange (Warning)
    "Infrastructure Gap / Access Constraint": "#c0392b",   # Deep Red (Critical)
    "Insufficient Data": "#ecf0f1",                        # Very Light Grey
    "Unknown": "#bdc3c7"
}

OUTPUT_FIG_DIR = Path("outputs/figures")
OUTPUT_TAB_DIR = Path("outputs/tables")

# Ensure directories exist
OUTPUT_FIG_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_TAB_DIR.mkdir(parents=True, exist_ok=True)


# =============================================================================
# PLOTTING FUNCTIONS
# =============================================================================

def plot_isi_distribution_enhanced(
    df: pd.DataFrame,
    isi_col: str = "isi_bounded",
    save_path: Optional[Path] = None
) -> None:
    """
    Plot ISI distribution with main focus on 0-0.4 and inset for critical tail.
    """
    if df.empty or isi_col not in df.columns:
        return

    fig, ax = plt.subplots(figsize=(12, 7))
    
    # Filter for valid data
    valid_isi = df[isi_col].dropna()
    
    # Main Plot (Clipped to 0.4 for visibility of bulk)
    # We plot the full histogram but set xlim to focus
    sns.histplot(
        valid_isi, bins=50, kde=False, 
        color="#34495e", edgecolor="white", alpha=0.9, ax=ax
    )
    
    # Shading Regions
    # Balanced (0 - 0.1)
    ax.axvspan(0, 0.1, color="#2ecc71", alpha=0.1, label="Balanced Zone")
    # Moderate (0.1 - 0.4)
    ax.axvspan(0.1, 0.4, color="#f1c40f", alpha=0.1, label="Moderate Lag")
    
    # Focus View Limits
    ax.set_xlim(0, 0.45)
    
    # Threshold Lines
    ax.axvline(0.1, color="#27ae60", linestyle=":", linewidth=2)
    ax.axvline(0.4, color="#f39c12", linestyle=":", linewidth=2)
    
    # Annotate Thresholds
    ax.text(0.05, ax.get_ylim()[1]*0.95, "Balanced", ha="center", color="#27ae60", fontweight="bold")
    ax.text(0.25, ax.get_ylim()[1]*0.95, "Moderate Lag", ha="center", color="#d35400", fontweight="bold")

    # Titles
    ax.set_title("Identity Staleness Index (ISI) Distribution (2025)\nFocus on Balanced & Moderate Zones", 
                 fontsize=16, pad=20, loc="left", fontweight="bold")
    ax.set_xlabel("ISI Score", fontsize=12)
    ax.set_ylabel("District Count", fontsize=12)
    
    # INSET: Critical Tail (0.4 - 1.0)
    # Position: Upper Right
    axins = inset_axes(ax, width="40%", height="40%", loc=1)
    
    # Plot only data > 0.3 for context in inset
    tail_data = valid_isi[valid_isi > 0.3]
    if not tail_data.empty:
        sns.histplot(tail_data, bins=20, color="#c0392b", edgecolor="white", ax=axins)
    
    axins.set_xlim(0.3, 1.0)
    axins.set_title("Critical Tail (ISI > 0.4)", fontsize=10, color="#c0392b", fontweight="bold")
    axins.axvline(0.7, color="#c0392b", linestyle="--", linewidth=1.5)
    axins.text(0.85, axins.get_ylim()[1]*0.8, "CRITICAL", ha="center", color="#c0392b", fontsize=8)
    
    # Annotation of Critical Count
    crit_count = (valid_isi > 0.7).sum()
    high_count = ((valid_isi > 0.4) & (valid_isi <= 0.7)).sum()
    
    plt.figtext(0.65, 0.55, 
                f"ACTION REQUIRED:\n{high_count} dist. in High Lag\n{crit_count} dist. in Critical", 
                fontsize=10, fontweight="bold", 
                bbox=dict(facecolor='white', edgecolor='#c0392b', boxstyle='round,pad=0.5'))

    sns.despine(ax=ax)
    sns.despine(ax=axins)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        logger.info(f"Saved Enhanced ISI plot to {save_path}")
    
    plt.close()


def plot_quadrant_diagnostic_enhanced(
    df: pd.DataFrame,
    isi_col: str = "isi_bounded",
    duv_col: str = "duv_score",
    quadrant_col: str = "diagnostic_quadrant",
    save_path: Optional[Path] = None
) -> None:
    """
    2x2 Quadrant Plot with explicit Action Zones.
    """
    if df.empty or not {isi_col, duv_col, quadrant_col}.issubset(df.columns):
        return

    plt.figure(figsize=(10, 10))
    plot_df = df[df[quadrant_col] != "Insufficient Data"].copy()
    
    # 1. Define Quadrant background regions
    # Quadrant 1: High ISI, Low DUV (Access Gap - Critical)
    plt.axvspan(ISI_HIGH_THRESHOLD, 1.0, ymin=0, ymax=0.5, color="#c0392b", alpha=0.1) # Red tint
    
    # Quadrant 2: High ISI, High DUV (Process Friction - Warning)
    plt.axvspan(ISI_HIGH_THRESHOLD, 1.0, ymin=0.5, ymax=1.0, color="#f1c40f", alpha=0.1) # Yellow tint
    
    # Quadrant 3 & 4 (Healthy/Low ISI) - Leave white or very light grey
    plt.axvspan(0, ISI_HIGH_THRESHOLD, color="#ecf0f1", alpha=0.2)
    
    # 2. Scatter Plot
    # Plot healthy points first (zorder lower)
    healthy_mask = plot_df[isi_col] <= ISI_HIGH_THRESHOLD
    critical_mask = plot_df[isi_col] > ISI_HIGH_THRESHOLD
    
    plt.scatter(
        plot_df[healthy_mask][isi_col], 
        plot_df[healthy_mask][duv_col],
        c="#95a5a6", alpha=0.4, s=50, label="Structurally Stable"
    )
    
    # Plot critical points (zorder higher, larger, edged)
    sns.scatterplot(
        data=plot_df[critical_mask],
        x=isi_col, y=duv_col,
        hue=quadrant_col,
        palette=QUADRANT_COLORS,
        s=100, edgecolor="black", linewidth=0.8, alpha=0.9,
        legend=False
    )
    
    # 3. Threshold Lines
    plt.axvline(ISI_HIGH_THRESHOLD, color="#7f8c8d", linestyle="--", linewidth=1.5)
    plt.axhline(DUV_HIGH_THRESHOLD, color="#7f8c8d", linestyle="--", linewidth=1.5)
    
    # 4. Quadrant Annotations (Fixed Positions relative to thresholds)
    # Using relative coordinates (transform=ax.transAxes for safety? No, data coords easier here due to fixed range mostly)
    # Actually, data coordinates are safer if we know range. Bounded ISI is [0,1]. DUV is unbounded, but usually < 1.0.
    
    # Top Right (High Friction)
    plt.text(0.7, max(DUV_HIGH_THRESHOLD * 2, 0.1), "PROCESS FRICTION\n(High Demand, Low throughput)", 
             fontsize=10, fontweight="bold", color="#d35400", ha="center")
    
    # Bottom Right (Access Gap)
    plt.text(0.7, DUV_HIGH_THRESHOLD / 2, "INFRASTRUCTURE GAP\n(Low Demand, Low Throughput)", 
             fontsize=10, fontweight="bold", color="#c0392b", ha="center")
    
    # Left Side (Stable)
    plt.text(ISI_HIGH_THRESHOLD / 2, max(DUV_HIGH_THRESHOLD * 2, 0.1), "DIGITALLY ENGAGED\n(Healthy)", 
             fontsize=10, fontweight="bold", color="#7f8c8d", ha="center", alpha=0.6)
    
    # 5. Counts
    counts = plot_df[quadrant_col].value_counts()
    # Manual placement of counts would be ideal, but let's put them in title or legend.
    # The user asked for "Annotate counts inside each quadrant".
    # I'll create a text box in corners.
    
    # Count Logic
    q_counts = {
        "Infra": len(plot_df[(plot_df[isi_col] > ISI_HIGH_THRESHOLD) & (plot_df[duv_col] <= DUV_HIGH_THRESHOLD)]),
        "Friction": len(plot_df[(plot_df[isi_col] > ISI_HIGH_THRESHOLD) & (plot_df[duv_col] > DUV_HIGH_THRESHOLD)]),
        "Healthy": len(plot_df[plot_df[isi_col] <= ISI_HIGH_THRESHOLD])
    }
    
    props = dict(boxstyle='round', facecolor='white', alpha=0.8)
    # Infra Count
    plt.text(0.95, 0.05, f"n={q_counts['Infra']}", transform=plt.gca().transAxes, 
             fontsize=12, fontweight="bold", color="#c0392b", bbox=props, ha="right")
    # Friction Count
    plt.text(0.95, 0.95, f"n={q_counts['Friction']}", transform=plt.gca().transAxes, 
             fontsize=12, fontweight="bold", color="#d35400", bbox=props, ha="right", va="top")
    
    
    # Titles
    plt.title("System Diagnostic Matrix (ISI × DUV)", fontsize=16, fontweight="bold", pad=20)
    plt.xlabel("Identity Staleness Index (ISI)", fontsize=12)
    plt.ylabel("Demographic Update Velocity (DUV)", fontsize=12)
    
    sns.despine()
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        logger.info(f"Saved Enhanced Quadrant plot to {save_path}")
    
    plt.close()


def plot_top_intervention_candidates(
    df: pd.DataFrame,
    save_path: Optional[Path] = None
) -> None:
    """
    Horizontal Bar Chart of Top 10 Districts by ISI.
    """
    if df.empty:
        return
        
    # Valid ISI only, descending
    top_10 = df.sort_values("isi_bounded", ascending=False).head(10).copy()
    
    if top_10.empty:
        return

    plt.figure(figsize=(10, 6))
    
    # Create colors list based on quadrant
    colors = [QUADRANT_COLORS.get(q, "#95a5a6") for q in top_10["diagnostic_quadrant"]]
    
    bars = plt.barh(
        y=np.arange(len(top_10)),
        width=top_10["isi_bounded"],
        color=colors,
        edgecolor="none",
        height=0.6
    )
    
    # Labels
    plt.yticks(np.arange(len(top_10)), top_10["district"] + ", " + top_10["state"])
    plt.gca().invert_yaxis() # Top at top
    
    # Value annotation
    for i, bar in enumerate(bars):
        val = bar.get_width()
        plt.text(val + 0.01, i, f"{val:.2f}", va="center", fontweight="bold", fontsize=10)
    
    # Threshold Line
    plt.axvline(0.7, color="#c0392b", linestyle="--", linewidth=1.5)
    plt.text(0.71, -0.8, "CRITICAL (0.7)", color="#c0392b", fontsize=10, fontweight="bold")
    
    plt.title("Immediate Intervention Candidates (2025)\nTop 10 Districts by Identity Staleness", 
              fontsize=14, fontweight="bold", loc="left", pad=15)
    
    plt.xlabel("ISI Score", fontsize=12)
    
    # Remove borders
    sns.despine(left=True, bottom=False)
    plt.grid(axis="x", alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        logger.info(f"Saved Action List plot to {save_path}")
    
    plt.close()


def generate_all_visuals(merged_df: pd.DataFrame):
    """Orchestrate decision-grade visualization."""
    
    logger.info("Generating decision-grade visualizations...")
    
    # 1. Enhanced ISI Distribution
    plot_isi_distribution_enhanced(
        merged_df, 
        save_path=OUTPUT_FIG_DIR / "isi_distribution_enhanced.png"
    )
    
    # 2. Enhanced Quadrant Scatter
    plot_quadrant_diagnostic_enhanced(
        merged_df, 
        save_path=OUTPUT_FIG_DIR / "isi_duv_quadrant_enhanced.png"
    )
    
    # 3. Action List
    plot_top_intervention_candidates(
        merged_df, 
        save_path=OUTPUT_FIG_DIR / "intervention_candidates.png"
    )
    
    logger.info("Visualizations complete.")


def generate_summary_tables(df: pd.DataFrame, output_dir: Path = OUTPUT_TAB_DIR):
    """Generate reporting tables (unchanged logic, ensures existence)."""
    # Simply reuse previous logic or minimal write
    if df.empty: return
    
    critical_mask = (df["risk_level"] == "CRITICAL") | (df["diagnostic_quadrant"].str.contains("Infrastructure"))
    critical_df = df[critical_mask].sort_values("isi_bounded", ascending=False)
    
    cols = ["state", "district", "isi_bounded", "duv_score", "risk_level", "diagnostic_quadrant"]
    critical_df[cols].head(50).to_csv(output_dir / "top_critical_districts.csv", index=False)
    
    summary = df["diagnostic_quadrant"].value_counts().reset_index()
    summary.columns = ["System State", "District Count"]
    summary["Percentage"] = (summary["District Count"] / summary["District Count"].sum() * 100).round(1)
    summary.to_csv(output_dir / "quadrant_summary.csv", index=False)
