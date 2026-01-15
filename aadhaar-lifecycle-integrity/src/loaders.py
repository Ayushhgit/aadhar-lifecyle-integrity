"""
Robust Data Loaders for UIDAI Aggregated Datasets.

Production-quality CSV loading functions with:
- Schema validation on load
- Safe type coercion
- Explicit handling of missing/malformed values
- Detailed logging of inconsistencies

Author: Principal Data Scientist
Constraints: Government-grade, policy-safe, deterministic
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
import logging
import glob

from .schemas import (
    ENROLMENT_SCHEMA,
    BIOMETRIC_UPDATE_SCHEMA,
    DEMOGRAPHIC_UPDATE_SCHEMA,
    ColumnSpec,
    FieldRequirement,
    get_mandatory_columns,
    get_datetime_columns,
    get_validation_rules,
)

# Configure module logger
logger = logging.getLogger(__name__)


@dataclass
class ValidationResult:
    """Result of schema validation."""
    is_valid: bool
    errors: List[str]
    warnings: List[str]
    rows_loaded: int
    rows_dropped: int
    
    def __str__(self) -> str:
        status = "VALID" if self.is_valid else "INVALID"
        return (
            f"ValidationResult({status}, "
            f"loaded={self.rows_loaded}, dropped={self.rows_dropped}, "
            f"errors={len(self.errors)}, warnings={len(self.warnings)})"
        )


class DataLoadError(Exception):
    """Raised when data loading fails critically."""
    pass


def _validate_columns(
    df: pd.DataFrame,
    schema: Dict[str, ColumnSpec],
    dataset_name: str
) -> Tuple[List[str], List[str]]:
    """
    Validate that DataFrame has required columns.
    
    Returns
    -------
    tuple
        (errors, warnings) lists
    """
    errors = []
    warnings = []
    
    expected_cols = set(schema.keys())
    actual_cols = set(df.columns)
    
    # Check for missing mandatory columns
    mandatory = {s.name for s in schema.values() if s.requirement == FieldRequirement.MANDATORY}
    missing_mandatory = mandatory - actual_cols
    if missing_mandatory:
        errors.append(f"[{dataset_name}] Missing mandatory columns: {missing_mandatory}")
    
    # Check for missing optional columns
    optional = {s.name for s in schema.values() if s.requirement == FieldRequirement.OPTIONAL}
    missing_optional = optional - actual_cols
    if missing_optional:
        warnings.append(f"[{dataset_name}] Missing optional columns (will use defaults): {missing_optional}")
    
    # Check for unexpected columns
    unexpected = actual_cols - expected_cols
    if unexpected:
        warnings.append(f"[{dataset_name}] Unexpected columns (will be ignored): {unexpected}")
    
    return errors, warnings


def _validate_values(
    df: pd.DataFrame,
    schema: Dict[str, ColumnSpec],
    dataset_name: str
) -> Tuple[List[str], List[str], pd.DataFrame]:
    """
    Validate column values against schema constraints.
    
    Returns
    -------
    tuple
        (errors, warnings, cleaned_df)
    """
    errors = []
    warnings = []
    df = df.copy()
    
    for col_name, spec in schema.items():
        if col_name not in df.columns:
            continue
        
        col = df[col_name]
        
        # Check for nulls in mandatory columns
        null_count = col.isnull().sum()
        if null_count > 0:
            if spec.requirement == FieldRequirement.MANDATORY:
                errors.append(
                    f"[{dataset_name}] Column '{col_name}' has {null_count} null values "
                    f"(mandatory field)"
                )
            else:
                warnings.append(
                    f"[{dataset_name}] Column '{col_name}' has {null_count} null values"
                )
        
        # Check numeric bounds
        if spec.min_value is not None and pd.api.types.is_numeric_dtype(col):
            violations = (col < spec.min_value).sum()
            if violations > 0:
                warnings.append(
                    f"[{dataset_name}] Column '{col_name}' has {violations} values "
                    f"below minimum ({spec.min_value})"
                )
        
        if spec.max_value is not None and pd.api.types.is_numeric_dtype(col):
            violations = (col > spec.max_value).sum()
            if violations > 0:
                warnings.append(
                    f"[{dataset_name}] Column '{col_name}' has {violations} values "
                    f"above maximum ({spec.max_value})"
                )
        
        # Check valid values (categorical)
        if spec.valid_values is not None:
            invalid = ~col.isin(spec.valid_values) & col.notna()
            invalid_count = invalid.sum()
            if invalid_count > 0:
                sample_invalid = col[invalid].unique()[:5]
                warnings.append(
                    f"[{dataset_name}] Column '{col_name}' has {invalid_count} invalid values. "
                    f"Sample: {list(sample_invalid)}"
                )
    
    return errors, warnings, df


def _coerce_dtypes(
    df: pd.DataFrame,
    schema: Dict[str, ColumnSpec],
    dataset_name: str,
    date_format: str = "%d-%m-%Y"
) -> Tuple[pd.DataFrame, List[str]]:
    """
    Safely coerce column data types according to schema.
    
    Returns
    -------
    tuple
        (coerced_df, warnings)
    """
    warnings = []
    df = df.copy()
    
    for col_name, spec in schema.items():
        if col_name not in df.columns:
            continue
        
        target_dtype = spec.dtype
        
        try:
            if target_dtype.startswith("datetime"):
                df[col_name] = pd.to_datetime(df[col_name], format=date_format, errors="coerce")
                null_after = df[col_name].isnull().sum()
                if null_after > 0:
                    warnings.append(
                        f"[{dataset_name}] Column '{col_name}': {null_after} values "
                        f"could not be parsed as datetime"
                    )
            
            elif target_dtype == "int64":
                # First convert to numeric, then to int (handling NaN)
                df[col_name] = pd.to_numeric(df[col_name], errors="coerce")
                if df[col_name].isnull().any():
                    warnings.append(
                        f"[{dataset_name}] Column '{col_name}': some values could not be "
                        f"converted to integer, will remain as float"
                    )
                else:
                    df[col_name] = df[col_name].astype("int64")
            
            elif target_dtype == "float64":
                df[col_name] = pd.to_numeric(df[col_name], errors="coerce")
            
            elif target_dtype == "str":
                df[col_name] = df[col_name].astype(str)
            
            elif target_dtype == "category":
                df[col_name] = df[col_name].astype("category")
        
        except Exception as e:
            warnings.append(
                f"[{dataset_name}] Column '{col_name}': dtype coercion failed - {str(e)}"
            )
    
    return df, warnings


def load_csv_files(
    directory: Path,
    pattern: str = "*.csv"
) -> pd.DataFrame:
    """
    Load and concatenate all CSV files matching pattern in directory.
    
    Parameters
    ----------
    directory : Path
        Directory containing CSV files.
    pattern : str
        Glob pattern for file matching.
        
    Returns
    -------
    pd.DataFrame
        Concatenated DataFrame from all matching files.
        
    Raises
    ------
    DataLoadError
        If no files found or all files fail to load.
    """
    directory = Path(directory)
    files = sorted(glob.glob(str(directory / pattern)))
    
    if not files:
        raise DataLoadError(f"No files matching '{pattern}' found in {directory}")
    
    logger.info(f"Found {len(files)} files in {directory}")
    
    dfs = []
    for filepath in files:
        try:
            df = pd.read_csv(filepath)
            df["_source_file"] = Path(filepath).name
            dfs.append(df)
            logger.debug(f"Loaded {len(df)} rows from {filepath}")
        except Exception as e:
            logger.error(f"Failed to load {filepath}: {e}")
    
    if not dfs:
        raise DataLoadError(f"All files in {directory} failed to load")
    
    combined = pd.concat(dfs, ignore_index=True)
    logger.info(f"Combined {len(combined)} total rows from {len(dfs)} files")
    
    return combined


def load_enrolment_data(
    data_dir: Optional[Path] = None,
    validate: bool = True
) -> Tuple[pd.DataFrame, ValidationResult]:
    """
    Load Aadhaar enrolment dataset with schema validation.
    
    Parameters
    ----------
    data_dir : Path, optional
        Directory containing enrolment CSV files.
        Defaults to data/raw/enrolment.
    validate : bool
        Whether to perform schema validation.
        
    Returns
    -------
    tuple
        (DataFrame, ValidationResult)
    """
    if data_dir is None:
        data_dir = Path(__file__).parent.parent / "data" / "raw" / "enrolment"
    
    dataset_name = "Enrolment"
    logger.info(f"Loading {dataset_name} data from {data_dir}")
    
    # Load raw data
    df = load_csv_files(data_dir)
    initial_rows = len(df)
    
    all_errors = []
    all_warnings = []
    
    if validate:
        # Column validation
        errors, warnings = _validate_columns(df, ENROLMENT_SCHEMA, dataset_name)
        all_errors.extend(errors)
        all_warnings.extend(warnings)
        
        # Type coercion
        df, coerce_warnings = _coerce_dtypes(df, ENROLMENT_SCHEMA, dataset_name)
        all_warnings.extend(coerce_warnings)
        
        # Value validation
        errors, warnings, df = _validate_values(df, ENROLMENT_SCHEMA, dataset_name)
        all_errors.extend(errors)
        all_warnings.extend(warnings)
    
    # Log all issues
    for error in all_errors:
        logger.error(error)
    for warning in all_warnings:
        logger.warning(warning)
    
    # Select only schema columns (+ source file for traceability)
    schema_cols = [c for c in ENROLMENT_SCHEMA.keys() if c in df.columns]
    schema_cols.append("_source_file")
    df = df[schema_cols]
    
    result = ValidationResult(
        is_valid=len(all_errors) == 0,
        errors=all_errors,
        warnings=all_warnings,
        rows_loaded=len(df),
        rows_dropped=initial_rows - len(df)
    )
    
    logger.info(f"{dataset_name} loading complete: {result}")
    
    return df, result


def load_biometric_updates(
    data_dir: Optional[Path] = None,
    validate: bool = True
) -> Tuple[pd.DataFrame, ValidationResult]:
    """
    Load Aadhaar biometric updates dataset with schema validation.
    
    Parameters
    ----------
    data_dir : Path, optional
        Directory containing biometric update CSV files.
        Defaults to data/raw/biometric_updates.
    validate : bool
        Whether to perform schema validation.
        
    Returns
    -------
    tuple
        (DataFrame, ValidationResult)
    """
    if data_dir is None:
        data_dir = Path(__file__).parent.parent / "data" / "raw" / "biometric_updates"
    
    dataset_name = "Biometric Updates"
    logger.info(f"Loading {dataset_name} data from {data_dir}")
    
    # Load raw data
    df = load_csv_files(data_dir)
    initial_rows = len(df)
    
    all_errors = []
    all_warnings = []
    
    if validate:
        # Column validation
        errors, warnings = _validate_columns(df, BIOMETRIC_UPDATE_SCHEMA, dataset_name)
        all_errors.extend(errors)
        all_warnings.extend(warnings)
        
        # Type coercion
        df, coerce_warnings = _coerce_dtypes(df, BIOMETRIC_UPDATE_SCHEMA, dataset_name)
        all_warnings.extend(coerce_warnings)
        
        # Value validation
        errors, warnings, df = _validate_values(df, BIOMETRIC_UPDATE_SCHEMA, dataset_name)
        all_errors.extend(errors)
        all_warnings.extend(warnings)
    
    # Log all issues
    for error in all_errors:
        logger.error(error)
    for warning in all_warnings:
        logger.warning(warning)
    
    # Select only schema columns
    schema_cols = [c for c in BIOMETRIC_UPDATE_SCHEMA.keys() if c in df.columns]
    schema_cols.append("_source_file")
    df = df[schema_cols]
    
    result = ValidationResult(
        is_valid=len(all_errors) == 0,
        errors=all_errors,
        warnings=all_warnings,
        rows_loaded=len(df),
        rows_dropped=initial_rows - len(df)
    )
    
    logger.info(f"{dataset_name} loading complete: {result}")
    
    return df, result


def load_demographic_updates(
    data_dir: Optional[Path] = None,
    validate: bool = True
) -> Tuple[pd.DataFrame, ValidationResult]:
    """
    Load Aadhaar demographic updates dataset with schema validation.
    
    Parameters
    ----------
    data_dir : Path, optional
        Directory containing demographic update CSV files.
        Defaults to data/raw/demographic_updates.
    validate : bool
        Whether to perform schema validation.
        
    Returns
    -------
    tuple
        (DataFrame, ValidationResult)
    """
    if data_dir is None:
        data_dir = Path(__file__).parent.parent / "data" / "raw" / "demographic_updates"
    
    dataset_name = "Demographic Updates"
    logger.info(f"Loading {dataset_name} data from {data_dir}")
    
    # Load raw data
    df = load_csv_files(data_dir)
    initial_rows = len(df)
    
    all_errors = []
    all_warnings = []
    
    if validate:
        # Column validation
        errors, warnings = _validate_columns(df, DEMOGRAPHIC_UPDATE_SCHEMA, dataset_name)
        all_errors.extend(errors)
        all_warnings.extend(warnings)
        
        # Type coercion
        df, coerce_warnings = _coerce_dtypes(df, DEMOGRAPHIC_UPDATE_SCHEMA, dataset_name)
        all_warnings.extend(coerce_warnings)
        
        # Value validation
        errors, warnings, df = _validate_values(df, DEMOGRAPHIC_UPDATE_SCHEMA, dataset_name)
        all_errors.extend(errors)
        all_warnings.extend(warnings)
    
    # Log all issues
    for error in all_errors:
        logger.error(error)
    for warning in all_warnings:
        logger.warning(warning)
    
    # Select only schema columns
    schema_cols = [c for c in DEMOGRAPHIC_UPDATE_SCHEMA.keys() if c in df.columns]
    schema_cols.append("_source_file")
    df = df[schema_cols]
    
    result = ValidationResult(
        is_valid=len(all_errors) == 0,
        errors=all_errors,
        warnings=all_warnings,
        rows_loaded=len(df),
        rows_dropped=initial_rows - len(df)
    )
    
    logger.info(f"{dataset_name} loading complete: {result}")
    
    return df, result


def load_all_datasets(
    base_dir: Optional[Path] = None,
    validate: bool = True
) -> Dict[str, Tuple[pd.DataFrame, ValidationResult]]:
    """
    Load all three datasets with validation.
    
    Parameters
    ----------
    base_dir : Path, optional
        Base directory containing data/raw subdirectories.
    validate : bool
        Whether to perform schema validation.
        
    Returns
    -------
    dict
        Dictionary mapping dataset name to (DataFrame, ValidationResult) tuple.
    """
    if base_dir is None:
        base_dir = Path(__file__).parent.parent / "data" / "raw"
    
    results = {
        "enrolment": load_enrolment_data(base_dir / "enrolment", validate),
        "biometric_updates": load_biometric_updates(base_dir / "biometric_updates", validate),
        "demographic_updates": load_demographic_updates(base_dir / "demographic_updates", validate),
    }
    
    # Summary
    total_rows = sum(r[1].rows_loaded for r in results.values())
    all_valid = all(r[1].is_valid for r in results.values())
    
    logger.info(f"All datasets loaded: {total_rows} total rows, all_valid={all_valid}")
    
    return results
