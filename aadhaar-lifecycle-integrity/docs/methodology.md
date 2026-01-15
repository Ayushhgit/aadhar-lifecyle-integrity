# Methodology

## Overview

This document outlines the methodology for analyzing Aadhaar lifecycle integrity through the computation of two key metrics: **Integrity Score Index (ISI)** and **Data Update Velocity (DUV)**.

## Data Sources

1. **Enrolment Data**: Initial Aadhaar enrolment records including biometric quality scores and verification status
2. **Biometric Updates**: Records of biometric data modifications (fingerprint, iris, face)
3. **Demographic Updates**: Records of demographic information changes (name, address, DOB)

## Integrity Score Index (ISI)

ISI is a composite metric measuring the overall integrity of an Aadhaar record.

### Components

| Component | Weight | Description |
|-----------|--------|-------------|
| Biometric Quality | 0.4 | Normalized biometric capture quality score |
| Update Frequency | 0.3 | Deviation from optimal update frequency |
| Consistency | 0.2 | Quality improvement trend across updates |
| Verification Rate | 0.1 | Success rate of verification attempts |

### Formula

```
ISI = w₁×BiometricQuality + w₂×UpdateFrequency + w₃×Consistency + w₄×VerificationRate
```

### Categories

- **HIGH** (≥0.8): Excellent integrity
- **MEDIUM** (0.5-0.8): Acceptable integrity  
- **LOW** (0.3-0.5): Needs attention
- **CRITICAL** (<0.3): Requires immediate review

## Data Update Velocity (DUV)

DUV measures the rate of updates over time.

### Calculation

```
DUV = (UpdateCount / TimeSpanDays) × 365
```

### Categories

- **DORMANT** (0): No updates
- **LOW** (<0.5): Below normal
- **NORMAL** (0.5-2.0): Expected range
- **HIGH** (2.0-5.0): Above normal
- **HYPERACTIVE** (>5.0): Requires investigation

## Joint Analysis

The combination of ISI and DUV creates a risk matrix for identifying records requiring attention.

## Cohort Analysis

Records are grouped by enrolment period to analyze temporal patterns and trends in data quality.
