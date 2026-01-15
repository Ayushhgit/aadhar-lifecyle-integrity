"""
Aadhaar Lifecycle Integrity Analysis Package.

This package provides tools for analyzing the integrity of Aadhaar
lifecycle data including enrolment, biometric updates, and demographic changes.

Core Modules:
- schemas: Data schema definitions for all datasets
- loaders: Robust CSV loading with validation
"""

__version__ = "1.0.0"
__author__ = "Principal Data Scientist"

# Core imports
from . import schemas
from . import loaders
from . import preprocess
from . import cohort_alignment
from . import risk_classification
from . import duv
from . import analysis
