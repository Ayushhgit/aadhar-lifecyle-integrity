# Assumptions

## Data Quality Assumptions

1. **Completeness**: Input data contains all mandatory fields as defined in the schema
2. **Accuracy**: Values in the source data accurately represent ground truth at time of capture
3. **Uniqueness**: Each Aadhaar number represents a unique individual

## Temporal Assumptions

1. **Date Validity**: All dates fall within expected ranges (enrolment after 2009, updates after enrolment)
2. **Sequential Ordering**: Updates occur chronologically after enrolment
3. **Observation Window**: Analysis uses a 365-day rolling window for velocity calculations

## Scoring Assumptions

1. **Linear Relationships**: Component scores can be combined linearly
2. **Weight Stability**: ISI weights remain constant across all cohorts
3. **Optimal Range**: DUV of 0.5-2.0 updates per year is considered normal

## Biometric Assumptions

1. **Quality Degradation**: Biometric quality may decrease over time due to aging
2. **Update Legitimacy**: Biometric updates represent genuine re-enrolment needs
3. **Score Comparability**: Quality scores are comparable across different capture devices

## Limitations

1. Data represents a sample and may not reflect the entire Aadhaar ecosystem
2. External factors (policy changes, seasonal variations) are not modeled
3. The analysis does not account for fraud detection beyond statistical anomalies
