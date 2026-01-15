
import logging
import sys
import pandas as pd
from pathlib import Path

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
sys.path.insert(0, '.')

from src.loaders import load_all_datasets
from src.preprocess import preprocess_all_datasets
from src.cohort_alignment import perform_cohort_alignment
from src.isi import compute_isi
from src.risk_classification import classify_staleness_risk
from src.duv import compute_duv
from src.analysis import perform_joint_analysis
from src.visuals import generate_all_visuals, generate_summary_tables

def run_reporting_pipeline():
    print("\n=== AADHAAR LIFECYCLE INTEGRITY: REPORT GENERATION ===\n")
    
    # 1. Load & Preprocess
    datasets = load_all_datasets()
    enrol, _ = datasets['enrolment']
    bio, _ = datasets['biometric_updates']
    demo, _ = datasets['demographic_updates']
    
    enrol_y, bio_y, demo_y, _ = preprocess_all_datasets(enrol, bio, demo)
    
    # 2. Compute ISI (District Level)
    aligned = perform_cohort_alignment(enrol_y, bio_y, target_year=2025, aggregation_level='district')
    isi_result = compute_isi(aligned.data)
    isi_classified = classify_staleness_risk(isi_result.data)
    
    # 3. Compute DUV
    duv_result = compute_duv(demo_y, enrol_y, target_year=2025, match_levels=['state', 'district'])
    
    # 4. Joint Analysis
    joint_result = perform_joint_analysis(isi_classified, duv_result.data, merge_keys=['state', 'district'])
    final_df = joint_result.data
    
    # 5. Generate Visuals & Tables
    print(f"Generating outputs for {len(final_df)} districts...")
    generate_all_visuals(final_df)
    generate_summary_tables(final_df)
    
    # 6. Narrative Generation
    # Calculate key stats for narrative
    total_districts = len(final_df)
    balanced_counts = final_df[final_df['diagnostic_quadrant'] == "Digitally Engaged & Balanced System"].shape[0]
    catch_up_counts = final_df[final_df['risk_level'] == "CATCH-UP"].shape[0]
    critical_counts = final_df[final_df['risk_level'] == "CRITICAL"].shape[0]
    
    narrative = f"""
AADHAAR LIFECYCLE INTEGRITY DIAGNOSTIC REPORT (2025)
====================================================

1. ISI FINDINGS (Cross-Sectional Identity Staleness Index)
----------------------------------------------------------
The 2025 administrative year is characterized by specific structural flow patterns:

*   **Dominance of 'Catch-up' Zones**: {catch_up_counts} districts exhibit update volumes exceeding their 
    entry cohort pressure (ISI < 0). This indicates active backlog clearance or high campaign efficacy, 
    structurally outperforming the baseline entry maintenance requirement.

*   **Critical Lag**: {critical_counts} districts remain in the Critical or Infrastructure Gap zone. 
    In these locations, maintenance volume is severely disconnected from entry pressure, suggesting 
    potential infrastructure accessibility constraints rather than resident non-compliance.

2. JOINT DIAGNOSTIC (ISI x DUV)
-------------------------------
*   **System State**: The vast majority ({balanced_counts}/{total_districts}) of districts classify as 
    "Digitally Engaged & Balanced". This quadrant represents a healthy system state where 
    biometric maintenance keeps pace with entry pressure (Low ISI) and the population 
    remains administratively active (High DUV).

*   **Friction Zones**: The small number of districts in the "High-Engagement, High-Friction" 
    quadrant suggests areas where the population is digitally active (updating mobile/address) 
    but facing specific bottlenecks in completing mandatory biometric updates.

Note: All diagnostics are derived from system throughput ratios. 
No individual resident behavior or migration is inferred.
"""
    
    # Save narrative
    output_text_path = Path("outputs/narrative_summary.txt")
    with open(output_text_path, "w") as f:
        f.write(narrative)
    
    print(f"\nNarrative summary saved to {output_text_path}")
    print(narrative)
    print("\n=== REPORT GENERATION COMPLETE ===")

if __name__ == "__main__":
    run_reporting_pipeline()
