"""
Configuration settings for the Aadhaar Lifecycle Integrity analysis.
"""

from pathlib import Path
from typing import Dict, Any
import yaml


# Base paths
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
INTERIM_DATA_DIR = DATA_DIR / "interim"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
FIGURES_DIR = OUTPUTS_DIR / "figures"
TABLES_DIR = OUTPUTS_DIR / "tables"

# Raw data file paths
ENROLMENT_FILE = RAW_DATA_DIR / "enrolment" / "aadhaar_enrolment.csv"
BIOMETRIC_UPDATES_FILE = RAW_DATA_DIR / "biometric_updates" / "aadhaar_biometric_updates.csv"
DEMOGRAPHIC_UPDATES_FILE = RAW_DATA_DIR / "demographic_updates" / "aadhaar_demographic_updates.csv"

# Data schema definitions
ENROLMENT_SCHEMA = {
    "enrolment_id": "str",
    "aadhaar_number": "str",
    "enrolment_date": "datetime64[ns]",
    "state_code": "str",
    "district_code": "str",
    "gender": "str",
    "age_group": "str",
    "enrolment_type": "str",
    "operator_id": "str",
    "enrolment_centre_id": "str",
    "biometric_quality_score": "float64",
    "document_type": "str",
    "verification_status": "str",
}

BIOMETRIC_UPDATE_SCHEMA = {
    "update_id": "str",
    "aadhaar_number": "str",
    "update_date": "datetime64[ns]",
    "update_type": "str",
    "previous_quality_score": "float64",
    "new_quality_score": "float64",
    "biometric_modality": "str",
    "operator_id": "str",
    "update_centre_id": "str",
    "reason_code": "str",
    "verification_status": "str",
}

DEMOGRAPHIC_UPDATE_SCHEMA = {
    "update_id": "str",
    "aadhaar_number": "str",
    "update_date": "datetime64[ns]",
    "field_updated": "str",
    "previous_value_hash": "str",
    "new_value_hash": "str",
    "operator_id": "str",
    "update_centre_id": "str",
    "document_type": "str",
    "verification_status": "str",
}

# Analysis parameters
ISI_WEIGHTS = {
    "biometric_quality": 0.4,
    "update_frequency": 0.3,
    "consistency": 0.2,
    "verification_success_rate": 0.1,
}

DUV_WINDOW_DAYS = 365  # Rolling window for DUV calculation
COHORT_LAG_DAYS = 30   # Lag period for cohort alignment

# Visualization settings
FIGURE_DPI = 300
FIGURE_FORMAT = "png"
COLOR_PALETTE = "viridis"


def load_config(config_path: Path = None) -> Dict[str, Any]:
    """
    Load configuration from a YAML file.
    
    Parameters
    ----------
    config_path : Path, optional
        Path to the configuration file.
        
    Returns
    -------
    dict
        Configuration dictionary.
    """
    if config_path is None:
        return {}
    
    with open(config_path, "r") as f:
        return yaml.safe_load(f)
