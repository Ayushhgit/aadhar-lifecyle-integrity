
import logging
import sys
import pandas as pd
import numpy as np

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

def run_joint_verification():
    print("\n=== JOINT DIAGNOSTIC VERIFICATION START ===\n")
    
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
    print(f"DUV Mean: {duv_result.metadata['mean_duv']:.3f}")
    
    # 4. Joint Analysis
    joint_result = perform_joint_analysis(isi_classified, duv_result.data, merge_keys=['state', 'district'])
    
    print("\n=== QUADRANT DISTRIBUTION ===")
    for q, count in joint_result.metadata['quadrant_distribution'].items():
        print(f"{q}: {count}")
    
    print("\nSample Output (First 5 Rows with Quadrants):")
    cols = ['state', 'district', 'isi_bounded', 'duv_score', 'diagnostic_quadrant']
    print(joint_result.data[cols].head().to_string())
    
    print("\n=== VERIFICATION COMPLETE ===")

if __name__ == "__main__":
    run_joint_verification()
