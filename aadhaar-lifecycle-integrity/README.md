# Aadhaar Lifecycle Integrity Analysis

A comprehensive analysis framework for understanding the lifecycle integrity of Aadhaar data, including enrolment patterns, biometric updates, and demographic changes.

## Project Structure

```
aadhaar-lifecycle-integrity/
├── data/
│   ├── raw/                    # Original, immutable data
│   ├── interim/                # Intermediate data transformations
│   └── processed/              # Final, canonical data sets
├── notebooks/                  # Jupyter notebooks for analysis
├── src/                        # Source code modules
├── outputs/
│   ├── figures/               # Generated graphics and figures
│   └── tables/                # Generated data tables
├── docs/                       # Documentation
└── requirements.txt           # Python dependencies
```

## Installation

```bash
pip install -r requirements.txt
```

## Notebooks

1. **01_data_loading_and_schema_validation.ipynb** - Load raw data and validate schemas
2. **02_cleaning_and_normalization.ipynb** - Data cleaning and normalization
3. **03_cohort_lag_alignment.ipynb** - Cohort analysis and lag alignment
4. **04_isi_computation.ipynb** - Integrity Score Index computation
5. **05_duv_computation.ipynb** - Data Update Velocity computation
6. **06_joint_isi_duv_analysis.ipynb** - Joint ISI-DUV analysis
7. **07_visualisations.ipynb** - Generate visualizations and reports

## Usage

Execute notebooks in sequence (01 → 07) to run the complete analysis pipeline.

## Documentation

- [Methodology](docs/methodology.md)
- [Assumptions](docs/assumptions.md)
- [Ethics and Compliance](docs/ethics_and_compliance.md)
