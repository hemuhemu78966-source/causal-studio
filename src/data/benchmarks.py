"""
Benchmark and Realistic Causal Dataset Generators.

Generates ground-truth datasets for:
1. Fintech: Loan Approval & Algorithmic Recourse
2. Healthcare: Confounded ICU Sepsis Treatment
3. E-Commerce: Churn Prevention & Discount Uplift
"""

import pandas as pd
import numpy as np
import os
from typing import Tuple, Dict, Any, Optional
import networkx as nx


def generate_credit_risk_dataset(n_samples: int = 2500, seed: int = 42) -> Tuple[pd.DataFrame, nx.DiGraph]:
    """
    Generate synthetic Credit Risk and Loan Approval causal dataset.

    Returns:
        (DataFrame, True Ground Truth DAG)
    """
    np.random.seed(seed)

    # 1. Age: 0=Young (<30), 1=Middle (30-50), 2=Senior (>50)
    age = np.random.choice([0, 1, 2], size=n_samples, p=[0.35, 0.45, 0.20])

    # 2. Income: 0=Low, 1=Medium, 2=High (influenced by Age)
    income = np.zeros(n_samples, dtype=int)
    for i in range(n_samples):
        if age[i] == 0:
            income[i] = np.random.choice([0, 1, 2], p=[0.6, 0.3, 0.1])
        elif age[i] == 1:
            income[i] = np.random.choice([0, 1, 2], p=[0.2, 0.5, 0.3])
        else:
            income[i] = np.random.choice([0, 1, 2], p=[0.3, 0.4, 0.3])

    # 3. Savings: 0=Low, 1=Medium, 2=High (influenced by Age and Income)
    savings = np.zeros(n_samples, dtype=int)
    for i in range(n_samples):
        prob_high = 0.15 * age[i] + 0.35 * income[i]
        prob_high = min(0.85, max(0.05, prob_high))
        prob_low = max(0.05, 0.8 - prob_high)
        prob_med = max(0.05, 1.0 - prob_high - prob_low)
        total = prob_low + prob_med + prob_high
        savings[i] = np.random.choice([0, 1, 2], p=[prob_low/total, prob_med/total, prob_high/total])

    # 4. Loan_Amount: 0=Small, 1=Medium, 2=Large (influenced by Income)
    loan_amount = np.zeros(n_samples, dtype=int)
    for i in range(n_samples):
        if income[i] == 0:
            loan_amount[i] = np.random.choice([0, 1, 2], p=[0.5, 0.35, 0.15])
        elif income[i] == 1:
            loan_amount[i] = np.random.choice([0, 1, 2], p=[0.25, 0.5, 0.25])
        else:
            loan_amount[i] = np.random.choice([0, 1, 2], p=[0.15, 0.35, 0.5])

    # 5. Credit_Score: 0=Poor, 1=Fair, 2=Excellent (influenced by Savings)
    credit_score = np.zeros(n_samples, dtype=int)
    for i in range(n_samples):
        if savings[i] == 0:
            credit_score[i] = np.random.choice([0, 1, 2], p=[0.65, 0.25, 0.10])
        elif savings[i] == 1:
            credit_score[i] = np.random.choice([0, 1, 2], p=[0.20, 0.55, 0.25])
        else:
            credit_score[i] = np.random.choice([0, 1, 2], p=[0.05, 0.30, 0.65])

    # 6. Installment_Rate: 0=Low, 1=High (influenced by Loan_Amount and Income)
    installment_rate = np.zeros(n_samples, dtype=int)
    for i in range(n_samples):
        burden = loan_amount[i] - income[i]
        p_high = 0.5 + 0.25 * burden
        p_high = min(0.9, max(0.1, p_high))
        installment_rate[i] = np.random.choice([0, 1], p=[1 - p_high, p_high])

    # 7. Approval: 0=Rejected, 1=Approved (Target influenced by Credit_Score, Savings, Installment_Rate)
    approval = np.zeros(n_samples, dtype=int)
    for i in range(n_samples):
        score_val = 0.4 * credit_score[i] + 0.35 * savings[i] - 0.4 * installment_rate[i]
        p_approved = 1.0 / (1.0 + np.exp(-2.5 * (score_val - 0.3)))
        p_approved = min(0.95, max(0.05, p_approved))
        approval[i] = np.random.choice([0, 1], p=[1 - p_approved, p_approved])

    df = pd.DataFrame({
        'Age': age,
        'Income': income,
        'Savings': savings,
        'Loan_Amount': loan_amount,
        'Credit_Score': credit_score,
        'Installment_Rate': installment_rate,
        'Approval': approval
    })

    # Ground truth DAG
    dag = nx.DiGraph()
    dag.add_edges_from([
        ('Age', 'Income'),
        ('Age', 'Savings'),
        ('Income', 'Savings'),
        ('Income', 'Loan_Amount'),
        ('Income', 'Installment_Rate'),
        ('Savings', 'Credit_Score'),
        ('Savings', 'Approval'),
        ('Loan_Amount', 'Installment_Rate'),
        ('Credit_Score', 'Approval'),
        ('Installment_Rate', 'Approval')
    ])

    return df, dag


def generate_healthcare_dataset(n_samples: int = 2500, seed: int = 42) -> Tuple[pd.DataFrame, nx.DiGraph]:
    """
    Generate synthetic Healthcare Clinical Trial & Sepsis treatment dataset.
    Exhibits confounding: severely ill patients are more likely to receive aggressive treatment.
    """
    np.random.seed(seed)

    # 1. Age: 0=Young (<45), 1=Elderly (>=45)
    age = np.random.choice([0, 1], size=n_samples, p=[0.5, 0.5])

    # 2. Disease_Severity: 0=Mild, 1=Severe (influenced by Age)
    disease_severity = np.zeros(n_samples, dtype=int)
    for i in range(n_samples):
        p_severe = 0.25 if age[i] == 0 else 0.70
        disease_severity[i] = np.random.choice([0, 1], p=[1 - p_severe, p_severe])

    # 3. Treatment: 0=Standard, 1=Aggressive (Confounded by Disease_Severity)
    treatment = np.zeros(n_samples, dtype=int)
    for i in range(n_samples):
        p_treat = 0.20 if disease_severity[i] == 0 else 0.85
        treatment[i] = np.random.choice([0, 1], p=[1 - p_treat, p_treat])

    # 4. Biomarker_A: 0=Normal, 1=Elevated (influenced by Disease_Severity)
    biomarker = np.zeros(n_samples, dtype=int)
    for i in range(n_samples):
        p_elevated = 0.15 if disease_severity[i] == 0 else 0.75
        biomarker[i] = np.random.choice([0, 1], p=[1 - p_elevated, p_elevated])

    # 5. Recovery: 0=Deteriorated, 1=Recovered (Target influenced by Treatment (+), Severity (-), Biomarker (-))
    recovery = np.zeros(n_samples, dtype=int)
    for i in range(n_samples):
        # Treatment actually helps (+0.45), but Severity hurts (-0.60)
        logit = 0.45 * treatment[i] - 0.60 * disease_severity[i] - 0.25 * biomarker[i] + 0.2
        p_rec = 1.0 / (1.0 + np.exp(-2.0 * logit))
        p_rec = min(0.95, max(0.05, p_rec))
        recovery[i] = np.random.choice([0, 1], p=[1 - p_rec, p_rec])

    df = pd.DataFrame({
        'Age': age,
        'Disease_Severity': disease_severity,
        'Treatment': treatment,
        'Biomarker_A': biomarker,
        'Recovery': recovery
    })

    dag = nx.DiGraph()
    dag.add_edges_from([
        ('Age', 'Disease_Severity'),
        ('Disease_Severity', 'Treatment'),
        ('Disease_Severity', 'Biomarker_A'),
        ('Disease_Severity', 'Recovery'),
        ('Treatment', 'Recovery'),
        ('Biomarker_A', 'Recovery')
    ])

    return df, dag


def generate_ecommerce_churn_dataset(n_samples: int = 2500, seed: int = 42) -> Tuple[pd.DataFrame, nx.DiGraph]:
    """
    Generate synthetic E-Commerce Customer Retention & Discount Uplift dataset.
    """
    np.random.seed(seed)

    # 1. Tenure: 0=New (<1yr), 1=Long-term (>=1yr)
    tenure = np.random.choice([0, 1], size=n_samples, p=[0.4, 0.6])

    # 2. Support_Calls: 0=Low, 1=High (New customers call more)
    support_calls = np.zeros(n_samples, dtype=int)
    for i in range(n_samples):
        p_high = 0.45 if tenure[i] == 0 else 0.20
        support_calls[i] = np.random.choice([0, 1], p=[1 - p_high, p_high])

    # 3. Monthly_Spend: 0=Low, 1=High (Long tenure spends more)
    monthly_spend = np.zeros(n_samples, dtype=int)
    for i in range(n_samples):
        p_spend = 0.30 if tenure[i] == 0 else 0.70
        monthly_spend[i] = np.random.choice([0, 1], p=[1 - p_spend, p_spend])

    # 4. Discount_Offered: 0=No, 1=Yes (Offered to high support calls or high spenders)
    discount = np.zeros(n_samples, dtype=int)
    for i in range(n_samples):
        p_disc = 0.2 + 0.4 * support_calls[i] + 0.3 * monthly_spend[i]
        p_disc = min(0.9, max(0.1, p_disc))
        discount[i] = np.random.choice([0, 1], p=[1 - p_disc, p_disc])

    # 5. Churn: 0=Stayed, 1=Churned (Target: Discount reduces churn, Support calls increase churn)
    churn = np.zeros(n_samples, dtype=int)
    for i in range(n_samples):
        risk = 0.6 * support_calls[i] - 0.55 * discount[i] - 0.25 * tenure[i] + 0.1
        p_churn = 1.0 / (1.0 + np.exp(-2.5 * risk))
        p_churn = min(0.95, max(0.05, p_churn))
        churn[i] = np.random.choice([0, 1], p=[1 - p_churn, p_churn])

    df = pd.DataFrame({
        'Tenure': tenure,
        'Support_Calls': support_calls,
        'Monthly_Spend': monthly_spend,
        'Discount_Offered': discount,
        'Churn': churn
    })

    dag = nx.DiGraph()
    dag.add_edges_from([
        ('Tenure', 'Support_Calls'),
        ('Tenure', 'Monthly_Spend'),
        ('Tenure', 'Churn'),
        ('Support_Calls', 'Discount_Offered'),
        ('Support_Calls', 'Churn'),
        ('Monthly_Spend', 'Discount_Offered'),
        ('Discount_Offered', 'Churn')
    ])

    return df, dag


def generate_education_dataset(n_samples: int = 2500, seed: int = 42) -> Tuple[pd.DataFrame, nx.DiGraph]:
    """
    Generate synthetic Education & Student Exam Performance dataset.
    """
    np.random.seed(seed)

    # 1. Parent_Education: 0=High School, 1=College, 2=Postgraduate
    parent_edu = np.random.choice([0, 1, 2], size=n_samples, p=[0.4, 0.4, 0.2])

    # 2. Attendance: 0=Low (<75%), 1=High (>=75%)
    attendance = np.zeros(n_samples, dtype=int)
    for i in range(n_samples):
        p_att = 0.5 + 0.2 * parent_edu[i]
        attendance[i] = np.random.choice([0, 1], p=[1 - p_att, p_att])

    # 3. Study_Hours: 0=Low (<10h/wk), 1=Medium (10-20h), 2=High (>20h)
    study_hours = np.zeros(n_samples, dtype=int)
    for i in range(n_samples):
        p_high = 0.15 + 0.25 * parent_edu[i] + 0.2 * attendance[i]
        p_high = min(0.8, max(0.1, p_high))
        p_low = max(0.1, 0.7 - p_high)
        p_med = max(0.1, 1.0 - p_high - p_low)
        total = p_low + p_med + p_high
        study_hours[i] = np.random.choice([0, 1, 2], p=[p_low/total, p_med/total, p_high/total])

    # 4. Tutoring: 0=No, 1=Yes (Offered to low attendance or high parent edu)
    tutoring = np.zeros(n_samples, dtype=int)
    for i in range(n_samples):
        p_tutor = 0.2 + 0.3 * (1 - attendance[i]) + 0.2 * parent_edu[i]
        p_tutor = min(0.85, max(0.1, p_tutor))
        tutoring[i] = np.random.choice([0, 1], p=[1 - p_tutor, p_tutor])

    # 5. Exam_Pass: 0=Failed, 1=Passed (Target influenced by Study_Hours, Tutoring, Attendance)
    exam_pass = np.zeros(n_samples, dtype=int)
    for i in range(n_samples):
        score = 0.45 * study_hours[i] + 0.35 * attendance[i] + 0.30 * tutoring[i] - 0.4
        p_pass = 1.0 / (1.0 + np.exp(-2.2 * score))
        p_pass = min(0.95, max(0.05, p_pass))
        exam_pass[i] = np.random.choice([0, 1], p=[1 - p_pass, p_pass])

    df = pd.DataFrame({
        'Parent_Education': parent_edu,
        'Attendance': attendance,
        'Study_Hours': study_hours,
        'Tutoring': tutoring,
        'Exam_Pass': exam_pass
    })

    dag = nx.DiGraph()
    dag.add_edges_from([
        ('Parent_Education', 'Attendance'),
        ('Parent_Education', 'Study_Hours'),
        ('Parent_Education', 'Tutoring'),
        ('Attendance', 'Study_Hours'),
        ('Attendance', 'Tutoring'),
        ('Attendance', 'Exam_Pass'),
        ('Study_Hours', 'Exam_Pass'),
        ('Tutoring', 'Exam_Pass')
    ])

    return df, dag


def generate_hr_attrition_dataset(n_samples: int = 2500, seed: int = 42) -> Tuple[pd.DataFrame, nx.DiGraph]:
    """
    Generate synthetic HR Employee Promotion & Retention dataset.
    """
    np.random.seed(seed)

    # 1. Experience_Level: 0=Junior (<3yr), 1=Mid (3-7yr), 2=Senior (>7yr)
    exp = np.random.choice([0, 1, 2], size=n_samples, p=[0.35, 0.45, 0.20])

    # 2. Overtime: 0=Normal, 1=High (Juniors do more overtime)
    overtime = np.zeros(n_samples, dtype=int)
    for i in range(n_samples):
        p_ot = 0.50 if exp[i] == 0 else 0.30 if exp[i] == 1 else 0.20
        overtime[i] = np.random.choice([0, 1], p=[1 - p_ot, p_ot])

    # 3. Training_Completed: 0=None, 1=Advanced (Mid/Senior do more training)
    training = np.zeros(n_samples, dtype=int)
    for i in range(n_samples):
        p_tr = 0.25 + 0.3 * exp[i]
        p_tr = min(0.85, max(0.15, p_tr))
        training[i] = np.random.choice([0, 1], p=[1 - p_tr, p_tr])

    # 4. Performance_Rating: 0=Low, 1=High (Influenced by Training and Overtime)
    perf = np.zeros(n_samples, dtype=int)
    for i in range(n_samples):
        p_perf = 0.3 + 0.35 * training[i] + 0.15 * (1 - overtime[i]) + 0.1 * exp[i]
        p_perf = min(0.9, max(0.1, p_perf))
        perf[i] = np.random.choice([0, 1], p=[1 - p_perf, p_perf])

    # 5. Promoted: 0=No, 1=Yes (Target: influenced by Performance, Experience, Training)
    promoted = np.zeros(n_samples, dtype=int)
    for i in range(n_samples):
        logit = 0.55 * perf[i] + 0.40 * exp[i] + 0.30 * training[i] - 0.7
        p_prom = 1.0 / (1.0 + np.exp(-2.5 * logit))
        p_prom = min(0.95, max(0.05, p_prom))
        promoted[i] = np.random.choice([0, 1], p=[1 - p_prom, p_prom])

    df = pd.DataFrame({
        'Experience_Level': exp,
        'Overtime': overtime,
        'Training_Completed': training,
        'Performance_Rating': perf,
        'Promoted': promoted
    })

    dag = nx.DiGraph()
    dag.add_edges_from([
        ('Experience_Level', 'Overtime'),
        ('Experience_Level', 'Training_Completed'),
        ('Experience_Level', 'Promoted'),
        ('Overtime', 'Performance_Rating'),
        ('Training_Completed', 'Performance_Rating'),
        ('Training_Completed', 'Promoted'),
        ('Performance_Rating', 'Promoted')
    ])

    return df, dag


def load_benchmark_dataset(domain: str = "fintech", n_samples: int = 2500, save_to_disk: bool = True) -> Tuple[pd.DataFrame, nx.DiGraph]:
    """
    Load or generate benchmark dataset by domain name ('fintech', 'healthcare', 'ecommerce', 'education', 'hr').
    """
    domain = domain.lower()
    if domain in ["fintech", "credit", "finance"]:
        df, dag = generate_credit_risk_dataset(n_samples=n_samples)
        filename = "fintech_credit_risk.csv"
    elif domain in ["healthcare", "medical", "sepsis"]:
        df, dag = generate_healthcare_dataset(n_samples=n_samples)
        filename = "healthcare_clinical.csv"
    elif domain in ["ecommerce", "churn", "marketing"]:
        df, dag = generate_ecommerce_churn_dataset(n_samples=n_samples)
        filename = "ecommerce_churn.csv"
    elif domain in ["education", "student", "school"]:
        df, dag = generate_education_dataset(n_samples=n_samples)
        filename = "education_performance.csv"
    elif domain in ["hr", "employee", "attrition", "promotion"]:
        df, dag = generate_hr_attrition_dataset(n_samples=n_samples)
        filename = "hr_promotion.csv"
    else:
        raise ValueError(f"Unknown domain '{domain}'. Choose from 'fintech', 'healthcare', 'ecommerce', 'education', 'hr'.")

    if save_to_disk:
        data_dir = os.path.join(os.path.dirname(__file__), "..", "..", "data")
        os.makedirs(data_dir, exist_ok=True)
        filepath = os.path.join(data_dir, filename)
        df.to_csv(filepath, index=False)

    return df, dag
