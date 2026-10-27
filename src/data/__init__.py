"""
Benchmark and Real-World Dataset Generators for Causal Decision Intelligence.
"""

from .benchmarks import (
    generate_credit_risk_dataset,
    generate_healthcare_dataset,
    generate_ecommerce_churn_dataset,
    generate_education_dataset,
    generate_hr_attrition_dataset,
    load_benchmark_dataset
)

__all__ = [
    "generate_credit_risk_dataset",
    "generate_healthcare_dataset",
    "generate_ecommerce_churn_dataset",
    "generate_education_dataset",
    "generate_hr_attrition_dataset",
    "load_benchmark_dataset"
]
