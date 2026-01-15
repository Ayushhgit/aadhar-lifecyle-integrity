
import logging
import sys
import pandas as pd
import numpy as np

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

# Add src to path
sys.path.insert(0, '.')

from src.loaders import load_all_datasets
from src.preprocess import preprocess_all_datasets
from src.cohort_alignment import perform_cohort_alignment
from src.isi import compute_isi
from src.risk_classification import classify_staleness_risk, generate_diagnostic_report

def run_pipeline():
    print("=== PIPELINE VERIFICATION START ===")
    
    # 1. Load Data
    datasets = load_all_datasets()
    enrol, _ = datasets['enrolment']
    bio, _ = datasets['biometric_updates']
    demo, _ = datasets['demographic_updates']
    
    # 2. Preprocess
    enrol_y, bio_y, demo_y, _ = preprocess_all_datasets(enrol, bio, demo)
    
    # 3. Cohort Alignment (Cross-Sectional)
    # Using district level for stability
    alignment_result = perform_cohort_alignment(
        enrol_y, bio_y, 
        target_year=2025, 
        phi=1.0, 
        aggregation_level='district'
    )
    
    print(f"\nAligned Data: {len(alignment_result.data)} districts aligned.")
    
    # 4. ISI Computation
    isi_result = compute_isi(alignment_result.data)
    print(f"ISI Mean (Bounded): {isi_result.metadata['mean_isi']:.3f}")
    
    # 5. Risk Classification
    risk_df = classify_staleness_risk(isi_result.data)
    
    # 6. Diagnostic Report
    report = generate_diagnostic_report(risk_df)
    
    print("\n=== DIAGNOSTIC REPORT ===")
    print(f"System Health Index: {report['system_health_index']:.3f}")
    print(f"Critical Units: {report['critical_units']}")
    print(f"Catch-up Units: {report['catch_up_units']}")
    print("\nRisk Distribution:")
    for k, v in report['risk_distribution'].items():
        print(f"  {k}: {v}")
        
    print("\nSample Output (First 5 rows):")
    cols = ['state', 'district', 'expected_updates', 'observed_updates', 'isi_raw', 'risk_level']
    print(risk_df[cols].head().to_string())
    
    print("\n=== PIPELINE VERIFICATION END ===")

if __name__ == "__main__":
    run_pipeline()
