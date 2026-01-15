"""
Data Schemas for UIDAI Aggregated Datasets.

Schemas derived from actual data inspection of raw CSV files.
All schemas are designed for aggregated, non-individual-level data only.

Data Characteristics (from inspection):
- Enrolment: 1,006,029 rows across 3 files
- Biometric Updates: 1,861,108 rows across 4 files  
- Demographic Updates: 2,071,700 rows across 5 files
- No null values in any dataset
- 52 unique states/UTs represented

Author: Principal Data Scientist
Constraints: Government-grade, policy-safe, deterministic
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Any
from enum import Enum


class FieldRequirement(Enum):
    """Field requirement levels."""
    MANDATORY = "mandatory"
    OPTIONAL = "optional"


@dataclass(frozen=True)
class ColumnSpec:
    """Specification for a single column."""
    name: str
    dtype: str
    requirement: FieldRequirement
    description: str
    valid_values: Optional[List[str]] = None
    min_value: Optional[float] = None
    max_value: Optional[float] = None


# =============================================================================
# SCHEMA 1: AADHAAR ENROLMENT DATASET
# =============================================================================
# Observed columns: date, state, district, pincode, age_0_5, age_5_17, age_18_greater
# Aggregation level: date × state × district × pincode
# =============================================================================

ENROLMENT_SCHEMA: Dict[str, ColumnSpec] = {
    "date": ColumnSpec(
        name="date",
        dtype="datetime64[ns]",
        requirement=FieldRequirement.MANDATORY,
        description="Date of enrolment activity (DD-MM-YYYY format in source)",
    ),
    "state": ColumnSpec(
        name="state",
        dtype="str",
        requirement=FieldRequirement.MANDATORY,
        description="State or Union Territory name",
    ),
    "district": ColumnSpec(
        name="district",
        dtype="str",
        requirement=FieldRequirement.MANDATORY,
        description="District name within the state",
    ),
    "pincode": ColumnSpec(
        name="pincode",
        dtype="int64",
        requirement=FieldRequirement.MANDATORY,
        description="6-digit postal index number",
        min_value=100000,
        max_value=999999,
    ),
    "age_0_5": ColumnSpec(
        name="age_0_5",
        dtype="int64",
        requirement=FieldRequirement.MANDATORY,
        description="Count of enrolments for age group 0-5 years (children)",
        min_value=0,
    ),
    "age_5_17": ColumnSpec(
        name="age_5_17",
        dtype="int64",
        requirement=FieldRequirement.MANDATORY,
        description="Count of enrolments for age group 5-17 years (minors)",
        min_value=0,
    ),
    "age_18_greater": ColumnSpec(
        name="age_18_greater",
        dtype="int64",
        requirement=FieldRequirement.MANDATORY,
        description="Count of enrolments for age group 18+ years (adults)",
        min_value=0,
    ),
}


# =============================================================================
# SCHEMA 2: AADHAAR BIOMETRIC UPDATE DATASET
# =============================================================================
# Observed columns: date, state, district, pincode, bio_age_5_17, bio_age_17_
# Aggregation level: date × state × district × pincode
# Note: Column 'bio_age_17_' appears to mean age >= 17
# =============================================================================

BIOMETRIC_UPDATE_SCHEMA: Dict[str, ColumnSpec] = {
    "date": ColumnSpec(
        name="date",
        dtype="datetime64[ns]",
        requirement=FieldRequirement.MANDATORY,
        description="Date of biometric update activity (DD-MM-YYYY format in source)",
    ),
    "state": ColumnSpec(
        name="state",
        dtype="str",
        requirement=FieldRequirement.MANDATORY,
        description="State or Union Territory name",
    ),
    "district": ColumnSpec(
        name="district",
        dtype="str",
        requirement=FieldRequirement.MANDATORY,
        description="District name within the state",
    ),
    "pincode": ColumnSpec(
        name="pincode",
        dtype="int64",
        requirement=FieldRequirement.MANDATORY,
        description="6-digit postal index number",
        min_value=100000,
        max_value=999999,
    ),
    "bio_age_5_17": ColumnSpec(
        name="bio_age_5_17",
        dtype="int64",
        requirement=FieldRequirement.MANDATORY,
        description="Count of biometric updates for age group 5-17 years",
        min_value=0,
    ),
    "bio_age_17_": ColumnSpec(
        name="bio_age_17_",
        dtype="int64",
        requirement=FieldRequirement.MANDATORY,
        description="Count of biometric updates for age group 17+ years",
        min_value=0,
    ),
}


# =============================================================================
# SCHEMA 3: AADHAAR DEMOGRAPHIC UPDATE DATASET
# =============================================================================
# Observed columns: date, state, district, pincode, demo_age_5_17, demo_age_17_
# Aggregation level: date × state × district × pincode
# Note: Column 'demo_age_17_' appears to mean age >= 17
# =============================================================================

DEMOGRAPHIC_UPDATE_SCHEMA: Dict[str, ColumnSpec] = {
    "date": ColumnSpec(
        name="date",
        dtype="datetime64[ns]",
        requirement=FieldRequirement.MANDATORY,
        description="Date of demographic update activity (DD-MM-YYYY format in source)",
    ),
    "state": ColumnSpec(
        name="state",
        dtype="str",
        requirement=FieldRequirement.MANDATORY,
        description="State or Union Territory name",
    ),
    "district": ColumnSpec(
        name="district",
        dtype="str",
        requirement=FieldRequirement.MANDATORY,
        description="District name within the state",
    ),
    "pincode": ColumnSpec(
        name="pincode",
        dtype="int64",
        requirement=FieldRequirement.MANDATORY,
        description="6-digit postal index number",
        min_value=100000,
        max_value=999999,
    ),
    "demo_age_5_17": ColumnSpec(
        name="demo_age_5_17",
        dtype="int64",
        requirement=FieldRequirement.MANDATORY,
        description="Count of demographic updates for age group 5-17 years",
        min_value=0,
    ),
    "demo_age_17_": ColumnSpec(
        name="demo_age_17_",
        dtype="int64",
        requirement=FieldRequirement.MANDATORY,
        description="Count of demographic updates for age group 17+ years",
        min_value=0,
    ),
}


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def get_mandatory_columns(schema: Dict[str, ColumnSpec]) -> List[str]:
    """Return list of mandatory column names from a schema."""
    return [
        spec.name for spec in schema.values()
        if spec.requirement == FieldRequirement.MANDATORY
    ]


def get_optional_columns(schema: Dict[str, ColumnSpec]) -> List[str]:
    """Return list of optional column names from a schema."""
    return [
        spec.name for spec in schema.values()
        if spec.requirement == FieldRequirement.OPTIONAL
    ]


def get_datetime_columns(schema: Dict[str, ColumnSpec]) -> List[str]:
    """Return list of columns that should be parsed as datetime."""
    return [
        spec.name for spec in schema.values()
        if spec.dtype.startswith("datetime")
    ]


def get_numeric_columns(schema: Dict[str, ColumnSpec]) -> List[str]:
    """Return list of numeric columns."""
    return [
        spec.name for spec in schema.values()
        if spec.dtype in ("int64", "float64")
    ]


def get_validation_rules(schema: Dict[str, ColumnSpec]) -> Dict[str, Dict[str, Any]]:
    """
    Return validation rules for each column.
    
    Returns
    -------
    dict
        Mapping of column name to validation rules dict.
    """
    rules = {}
    for spec in schema.values():
        col_rules = {"dtype": spec.dtype, "required": spec.requirement == FieldRequirement.MANDATORY}
        if spec.valid_values:
            col_rules["valid_values"] = spec.valid_values
        if spec.min_value is not None:
            col_rules["min_value"] = spec.min_value
        if spec.max_value is not None:
            col_rules["max_value"] = spec.max_value
        rules[spec.name] = col_rules
    return rules
