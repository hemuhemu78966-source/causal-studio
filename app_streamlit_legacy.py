"""
Causal Studio - Interactive Causal Discovery, Intervention & Algorithmic Recourse Platform.
"""

import streamlit as st
import pandas as pd
import numpy as np
import networkx as nx
import plotly.express as px
import plotly.graph_objects as go
import os
import sys
import tempfile
from pyvis.network import Network
import streamlit.components.v1 as components

# Add src to path
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

from src.models.bayesian_network import CausalBayesianNetwork
from src.models.parameter_learning import ParameterLearning
from src.discovery.structure_learning import CausalDiscovery
from src.inference.intervention import InterventionAnalysis
from src.recourse.recourse_optimizer import AlgorithmicRecourse
from src.data.benchmarks import load_benchmark_dataset, generate_credit_risk_dataset, generate_healthcare_dataset, generate_ecommerce_churn_dataset
from src.utils.helpers import compute_effect_size_ci


st.set_page_config(
    page_title="Causal Studio | Decision Intelligence",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1E3A8A;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.1rem;
        color: #4B5563;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background: #F8FAFC;
        border-radius: 8px;
        padding: 16px;
        border: 1px solid #E2E8F0;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    }
    .recourse-card {
        background: #EFF6FF;
        border-left: 5px solid #3B82F6;
        padding: 16px;
        border-radius: 6px;
        margin-bottom: 12px;
    }
    .recourse-title {
        font-size: 1.1rem;
        font-weight: 600;
        color: #1E40AF;
    }
</style>
""", unsafe_allow_html=True)


# ==============================================================================
# SIDEBAR: Data Ingestion & Configuration
# ==============================================================================
st.sidebar.image("https://img.icons8.com/fluency/96/artificial-intelligence.png", width=64)
st.sidebar.title("Causal Studio")
st.sidebar.caption("Causal AI & Decision Intelligence Engine")

st.sidebar.markdown("---")
st.sidebar.subheader("1. Data Source")

data_mode = st.sidebar.selectbox(
    "Choose Dataset Mode:",
    [
        "Benchmark: Fintech Credit Risk",
        "Benchmark: Healthcare Clinical",
        "Benchmark: E-Commerce Churn",
        "Benchmark: Education Student Exam",
        "Benchmark: HR Employee Promotion",
        "Upload Custom CSV"
    ]
)

ground_truth_dag = None

if data_mode == "Benchmark: Fintech Credit Risk":
    df, ground_truth_dag = generate_credit_risk_dataset(n_samples=2000)
    domain_label = "Fintech Credit Risk"
    default_target = "Approval"
    default_desired = 1
elif data_mode == "Benchmark: Healthcare Clinical":
    df, ground_truth_dag = generate_healthcare_dataset(n_samples=2000)
    domain_label = "Healthcare Sepsis / Clinical"
    default_target = "Recovery"
    default_desired = 1
elif data_mode == "Benchmark: E-Commerce Churn":
    df, ground_truth_dag = generate_ecommerce_churn_dataset(n_samples=2000)
    domain_label = "E-Commerce Customer Churn"
    default_target = "Churn"
    default_desired = 0
elif data_mode == "Benchmark: Education Student Exam":
    from src.data.benchmarks import generate_education_dataset
    df, ground_truth_dag = generate_education_dataset(n_samples=2000)
    domain_label = "Education & Student Performance"
    default_target = "Exam_Pass"
    default_desired = 1
elif data_mode == "Benchmark: HR Employee Promotion":
    from src.data.benchmarks import generate_hr_attrition_dataset
    df, ground_truth_dag = generate_hr_attrition_dataset(n_samples=2000)
    domain_label = "HR Employee Promotion"
    default_target = "Promoted"
    default_desired = 1
else:
    uploaded_file = st.sidebar.file_uploader("Upload CSV Dataset", type=["csv"])
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        # Discretize any continuous float columns with >10 unique values for Bayesian Tabular learning
        for col in df.select_dtypes(include=[np.number]).columns:
            if df[col].nunique() > 5:
                df[col] = pd.qcut(df[col], q=3, labels=False, duplicates='drop')
        domain_label = "Custom Uploaded Dataset"
        default_target = df.columns[-1]
        default_desired = int(df[default_target].unique()[0])
    else:
        st.info("Please upload a CSV file or select a benchmark dataset from the sidebar.")
        st.stop()

st.sidebar.markdown("---")
st.sidebar.subheader("2. Causal Discovery Engine")

algorithm = st.sidebar.selectbox(
    "Structure Learning Algorithm:",
    ["Score-Based: Hill-Climbing (BIC Score)", "Constraint-Based: PC Algorithm", "Tree-Augmented Naive Bayes (TAN)"]
)

learn_button = st.sidebar.button("🚀 Re-Learn Causal Graph", use_container_width=True)


# ==============================================================================
# SESSION STATE / CACHED MODEL
# ==============================================================================
@st.cache_resource(show_spinner=False)
def get_or_fit_model(dataset_tuple_hash, algo_name):
    discovery = CausalDiscovery(df)
    if "Hill-Climbing" in algo_name:
        cbn = discovery.discover_hill_climb(scoring_method="bic")
    elif "PC Algorithm" in algo_name:
        cbn = discovery.discover_pc(ci_test="chi_square", significance_level=0.05)
    else:
        cbn = discovery.discover_tree(class_node=default_target, estimator_type="tan")

    # Fit parameters with Bayesian estimation (avoids zero division)
    param_learner = ParameterLearning(cbn)
    param_learner.fit_bayesian(df, prior_type="BDeu", equivalent_sample_size=10)
    metrics = discovery.evaluate_structure(true_dag=ground_truth_dag)
    return cbn, metrics


# Create a hashable key for caching
data_key = f"{domain_label}_{algorithm}_{len(df)}"
with st.spinner("Learning Causal DAG and Fitting Conditional Distributions..."):
    cbn, discovery_metrics = get_or_fit_model(data_key, algorithm)


# ==============================================================================
# MAIN PAGE HEADER & TABS
# ==============================================================================
st.markdown('<div class="main-header">🧠 Causal Studio & Decision Intelligence</div>', unsafe_allow_html=True)
st.markdown(f'<div class="sub-header">Active Domain: <b>{domain_label}</b> | Sample Count: <b>{len(df):,} rows</b> | Variables: <b>{len(df.columns)}</b></div>', unsafe_allow_html=True)

tab1, tab2, tab3, tab4 = st.tabs([
    "🌐 1. Causal Discovery & Graph",
    "⚡ 2. Do-Calculus & Interventions",
    "🎯 3. Algorithmic Recourse Optimizer",
    "📊 4. Dataset & Probability Explorer"
])


# ==============================================================================
# TAB 1: Causal Discovery & Graph
# ==============================================================================
with tab1:
    col_left, col_right = st.columns([2, 1])

    with col_left:
        st.subheader("Discovered Causal Directed Acyclic Graph (DAG)")
        st.caption("Interactive causal diagram: drag nodes, zoom, and inspect causal flow direction.")

        # Build PyVis Network
        net = Network(height="450px", width="100%", directed=True, bgcolor="#FFFFFF", font_color="#1E293B")
        net.barnes_hut(gravity=-3000, central_gravity=0.3, spring_length=120)

        for node in cbn.dag.nodes():
            is_target = (node == default_target)
            color = "#EF4444" if is_target else "#3B82F6"
            net.add_node(node, label=node, color=color, size=24 if is_target else 18, shape="dot", font={"size": 14, "face": "Arial"})

        for u, v in cbn.dag.edges():
            net.add_edge(u, v, color="#94A3B8", width=2, arrows="to")

        # Save to temp html and render
        with tempfile.NamedTemporaryFile(delete=False, suffix=".html") as tmp:
            net.save_graph(tmp.name)
            with open(tmp.name, 'r', encoding='utf-8') as f:
                html_content = f.read()
            components.html(html_content, height=470)

    with col_right:
        st.subheader("Discovery Metrics")
        
        st.metric("Learned Causal Edges", cbn.dag.number_of_edges())
        st.metric("BIC Model Score", f"{discovery_metrics.get('bic_score', 0):,.1f}")

        if ground_truth_dag:
            st.markdown("#### Comparison vs Ground Truth")
            shd = discovery_metrics.get('structural_hamming_distance', 'N/A')
            precision = discovery_metrics.get('precision', 0.0) * 100
            recall = discovery_metrics.get('recall', 0.0) * 100
            f1 = discovery_metrics.get('f1_score', 0.0) * 100

            st.metric("Structural Hamming Dist (SHD)", f"{shd} (lower is better)")
            col_m1, col_m2 = st.columns(2)
            col_m1.metric("Precision", f"{precision:.1f}%")
            col_m2.metric("Recall", f"{recall:.1f}%")
            st.metric("Graph F1 Score", f"{f1:.1f}%")


# ==============================================================================
# TAB 2: Do-Calculus & Interventions
# ==============================================================================
with tab2:
    st.subheader("Level 2 Causality: Simulating Interventions via do-calculus")
    st.caption("Manipulating variable X directly breaks its incoming confounding arrows: P(Y | do(X=x)).")

    col_int1, col_int2 = st.columns([1, 2])

    intervention_engine = InterventionAnalysis(cbn)
    all_vars = list(cbn.dag.nodes())

    with col_int1:
        st.markdown("#### Configure Intervention")
        target_var = st.selectbox("Target Outcome Variable (Y):", all_vars, index=all_vars.index(default_target) if default_target in all_vars else len(all_vars)-1)
        
        candidate_treatments = [v for v in all_vars if v != target_var]
        treatment_var = st.selectbox("Intervention Variable (X):", candidate_treatments, index=0)

        # Get unique values for treatment
        treatment_states = sorted(list(df[treatment_var].unique()))
        treat_val = st.selectbox(f"Set Intervention value do({treatment_var} = x):", treatment_states, index=len(treatment_states)-1)

        # Backdoor path checking
        backdoor_paths = intervention_engine._find_backdoor_paths(treatment_var, target_var)
        adjustment_set = intervention_engine.find_adjustment_set(treatment_var, target_var)

        st.markdown("---")
        st.markdown("#### Confounder Adjustment")
        if backdoor_paths:
            st.warning(f"⚠️ Found {len(backdoor_paths)} confounding backdoor path(s)!")
            if adjustment_set is not None:
                st.info(f"✅ Minimal Adjustment Set to Block Confounding: **{list(adjustment_set) if adjustment_set else 'None needed'}**")
        else:
            st.success("✅ No backdoor confounding paths detected between treatment and outcome.")

    with col_int2:
        st.markdown("#### Interventional vs Observational Shift")

        # 1. Observational distribution P(Y | X=x)
        obs_dist = {}
        target_card = cbn.cpds[target_var].variable_card if target_var in cbn.cpds else 2
        
        # Calculate interventional distribution P(Y | do(X=x))
        int_dist = intervention_engine.do_intervention({treatment_var: treat_val}, target_var)

        # Calculate baseline observational
        subset = df[df[treatment_var] == treat_val]
        if len(subset) > 0:
            obs_counts = subset[target_var].value_counts(normalize=True).to_dict()
            obs_dist = {str(k): obs_counts.get(k, 0.0) for k in range(target_card)}
        else:
            obs_dist = int_dist

        states = sorted(list(set(list(int_dist.keys()) + list(obs_dist.keys()))))
        chart_df = pd.DataFrame({
            "Outcome State": states * 2,
            "Probability": [obs_dist.get(s, 0.0) for s in states] + [int_dist.get(s, 0.0) for s in states],
            "Type": ["Observational P(Y|X=x)"] * len(states) + [f"Interventional P(Y|do(X={treat_val}))"] * len(states)
        })

        fig = px.bar(
            chart_df,
            x="Outcome State",
            y="Probability",
            color="Type",
            barmode="group",
            title=f"Causal Shift for {target_var} under do({treatment_var}={treat_val})",
            color_discrete_sequence=["#94A3B8", "#3B82F6"]
        )
        st.plotly_chart(fig, use_container_width=True)

        # Average Treatment Effect
        if len(treatment_states) >= 2:
            ate = intervention_engine.average_treatment_effect(treatment_var, target_var)
            st.metric(
                label=f"Average Treatment Effect (ATE) of {treatment_var} on {target_var}",
                value=f"{ate:+.3f}",
                help="ATE = E[Y | do(T=1)] - E[Y | do(T=0)]"
            )


# ==============================================================================
# TAB 3: Algorithmic Recourse Optimizer
# ==============================================================================
with tab3:
    st.subheader("Level 3 Causality: Prescriptive Algorithmic Recourse")
    st.caption("Calculate the minimum-effort set of causal actions an individual or organization should take to achieve a desired outcome.")

    col_rec1, col_rec2 = st.columns([1, 2])

    with col_rec1:
        st.markdown("#### 1. Select Profile to Recourse")
        row_idx = st.number_input("Select Individual Row from Dataset:", min_value=0, max_value=len(df)-1, value=0, step=1)
        factual_row = df.iloc[row_idx].to_dict()

        st.json(factual_row)

        st.markdown("#### 2. Recourse Goal")
        target_r = st.selectbox("Goal Outcome Variable:", all_vars, index=all_vars.index(default_target) if default_target in all_vars else len(all_vars)-1, key="rec_tgt")
        target_states = sorted(list(df[target_r].unique()))
        desired_val = st.selectbox("Desired Target Value:", target_states, index=len(target_states)-1, key="rec_des")

        current_val = factual_row.get(target_r, "N/A")
        st.write(f"Current State: **{target_r} = {current_val}** ➔ Desired: **{target_r} = {desired_val}**")

        st.markdown("#### 3. Constraints")
        all_features = [c for c in df.columns if c != target_r]
        immutable_vars = st.multiselect("Immutable Features (Cannot be altered):", all_features, default=["Age"] if "Age" in all_features else [])

        conf_threshold = st.slider("Target Confidence Threshold:", min_value=0.50, max_value=0.99, value=0.65, step=0.05)

    with col_rec2:
        st.markdown("#### Recommended Recourse Action Plans")

        if st.button("✨ Compute Minimum-Cost Action Plan", type="primary", use_container_width=True):
            optimizer = AlgorithmicRecourse(
                network=cbn,
                immutable_features=immutable_vars
            )

            with st.spinner("Optimizing causal action paths..."):
                plans = optimizer.compute_recourse(
                    factual_instance=factual_row,
                    target_variable=target_r,
                    desired_state=desired_val,
                    threshold=conf_threshold,
                    max_interventions=3,
                    top_k=3
                )

            if plans:
                st.success(f"Found {len(plans)} viable causal recourse pathway(s)!")

                for idx, plan in enumerate(plans, 1):
                    with st.container():
                        st.markdown(f"""
                        <div class="recourse-card">
                            <div class="recourse-title">Option #{idx}: Total Action Cost = {plan.total_cost:.2f}</div>
                            <p style="margin-top: 6px; font-size: 1.05rem;">{plan.explanation}</p>
                        </div>
                        """, unsafe_allow_html=True)

                        c1, c2, c3 = st.columns(3)
                        c1.metric("Baseline Success Probability", f"{plan.baseline_probability*100:.1f}%")
                        c2.metric("Post-Action Success Probability", f"{plan.target_probability*100:.1f}%")
                        c3.metric("Absolute Gain", f"+{(plan.target_probability - plan.baseline_probability)*100:.1f}%")

                        # Display tabular breakdown of changes
                        change_table = []
                        for f_name, change in plan.feature_changes.items():
                            change_table.append({
                                "Feature": f_name,
                                "Current Value": change["from"],
                                "Required Action Value": change["to"],
                                "Direction": "Increase" if change["to"] > change["from"] else "Decrease" if isinstance(change["to"], (int, float)) and isinstance(change["from"], (int, float)) and change["to"] < change["from"] else "Change"
                            })
                        st.dataframe(pd.DataFrame(change_table), use_container_width=True)
                        st.markdown("---")
            else:
                st.error("No valid recourse plan found within the given constraints and confidence threshold. Try lowering the confidence threshold or freeing up immutable features.")


# ==============================================================================
# TAB 4: Dataset & Probability Explorer
# ==============================================================================
with tab4:
    st.subheader("Dataset Preview & Distribution Summaries")
    st.dataframe(df.head(50), use_container_width=True)

    col_d1, col_d2 = st.columns(2)
    with col_d1:
        sel_col = st.selectbox("Select Column for Distribution:", df.columns)
        fig_dist = px.histogram(df, x=sel_col, title=f"Distribution of {sel_col}", color_discrete_sequence=["#3B82F6"])
        st.plotly_chart(fig_dist, use_container_width=True)

    with col_d2:
        st.markdown("#### Conditional Probability Distribution (CPD) Table")
        if sel_col in cbn.cpds:
            cpd_obj = cbn.cpds[sel_col]
            st.text(str(cpd_obj))
        else:
            st.info("No CPD available for this variable.")
