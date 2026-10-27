# Demonstrating Causal Studio 2.0

## 1. Understand the project in one minute

**Project title:** Causal Studio: Explainable Decision Intelligence with Bayesian Networks.

**Problem:** A predictive model can estimate an outcome, but a user also wants to explore what could happen after taking an action. Our application brings prediction, graph inspection, intervention simulation and constrained action planning into one research workflow.

**Opening script:**

> Good morning. Our project is Causal Studio, an explainable AI application based on Bayesian networks. It learns probabilistic relationships from tabular data, visualizes the graph, simulates interventions, and searches for low-cost actions under constraints. We evaluate predictions on unseen test records and make the causal assumptions visible.

This is probabilistic AI. It does not need a chatbot or a deep neural network to qualify as an AI project.

## 2. Before the evaluator arrives

1. Double-click `Launch Causal Studio.bat` in the project folder. Keep its terminal open.
2. Open http://127.0.0.1:8000 and wait until the overview appears.
3. Choose **Configure model**. Select **Credit intelligence**, **Hill-Climbing · BIC**, **2,000** records, seed **42**. Choose **Build workspace**.
4. Keep browser zoom at 100%. A wide desktop display shows the dashboard best; mobile is also supported.
5. Use **Export report** to save a snapshot as backup. Save a scenario and an action plan if needed.
6. Read the eight steps in the app's **Demo guide** once before your presentation.

Do not memorize a result you have not checked. Different seeds, data or algorithms change the numbers.

## 3. A seven-to-ten-minute live demonstration

### Minute 0–1: explain the problem and dashboard

Show **Overview**. Say:

> The dashboard summarizes the actual dataset and fitted model. We have 2,000 generated records. We reserve 1,600 for training and 400 for testing. We display learned connections, predictive accuracy and model integrity separately.

For the tested default configuration, Bayesian accuracy is **77.0%**, random forest accuracy is **75.25%**, and the majority baseline is **59.75%**. The Bayesian model improves on the majority baseline by **17.25 percentage points**. These are single-split results on synthetic data, not real-world performance guarantees.

The integrity indicator means the graph is acyclic and its probability tables pass validation. It does not certify causal correctness.

### Minute 1–2: explain the data

Open **Dataset**.

> The credit dataset contains age, income, savings, loan amount, credit score, installment rate and approval. We use synthetic data so we know the mechanism that generated it and have a graph against which to compare discovery.

Show readable category labels and a distribution. Search for `Approved`. Download the CSV if asked.

Explain the preprocessing:

- The train/test split happens first.
- Categorical states are encoded consistently from training data.
- Continuous custom features are discretized using training-only quantile cut points.
- The test data does not participate in graph discovery or parameter fitting.

A real CSV can be uploaded through **Configure model → Your dataset**. It must have enough examples per outcome state. Missing values, rare test-only categories and overly complex graphs receive explanatory errors. The app cannot invent causal ground truth for real data.

### Minute 2–3: show the graph

Open **Causal graph**. Drag a node, click to focus its neighboring connections, and use zoom/reset. Then briefly switch to **Known benchmark graph** and back.

> Each node is a variable. The learned arrows describe a proposed directed dependency structure. The lime node is our chosen outcome. Hill-Climbing searches over candidate graphs using BIC, balancing fit against complexity. We then fit smoothed conditional probability tables using Bayesian estimation.

**Critical point:** The learned graph may contain reversed arrows. Observational data can support multiple equivalent graphs. Do not describe every learned arrow as proven real-world causation. The known graph comes from the synthetic generator; showing it is a comparison, not a learned achievement.

If asked why the two graphs differ:

> Statistical uncertainty, equivalent structures and the learning assumptions can produce different directions. This is why we evaluate structure separately and would use domain-expert knowledge for a real application.

### Minute 3–4: simulate an intervention

Open **Interventions**. On the default credit model:

1. Select **Savings**.
2. Set **Current / control setting → Low**.
3. Set **Action setting → High**.
4. Choose **Run simulation**.
5. Show the probability bars and incoming-edge display.

Say:

> Observing high savings asks about existing people with that value. Intervening means forcing savings to that state in our model. The engine removes incoming arrows into the intervention variable, replaces its probability table with a point mass, and propagates the change.

In the tested default model, approval probability changes from approximately **30.0%** under low savings to **85.9%** under high savings: about **55.8 percentage points**. It is a scenario estimate under the learned graph. Savings has no learned parents in this particular run, so observation and intervention can be identical. That equality is a legitimate result, not a broken simulation.

Do not say this guarantees an applicant's real approval or identifies their personal counterfactual.

**Optional confounding example:** Configure **Clinical outcomes** with **Known benchmark graph**. Intervene on **Treatment** from **Standard** to **Aggressive**. The graph explicitly includes severity affecting both treatment and recovery. Explain why model-conditioned observation can differ from intervention. State that you deliberately selected the known generator graph for this demonstration; it was not discovered.

### Minute 4–6: search for actions

Return to **Credit intelligence / Hill-Climbing / 2,000 / seed 42** if you changed configuration. Open **Action planner**.

1. Use the first sample profile.
2. Set **Savings → Low** and **Credit Score → Poor** to demonstrate an unfavorable setting.
3. Keep **Age** locked under **Immutable features & action costs**.
4. Keep the probability threshold at **65%** and maximum changes at **2**.
5. Choose **Find action plans**.
6. Show each proposed feature change, modeled probability, baseline and normalized cost.
7. Export the result if asked.

Say:

> The optimizer searches subsets of actionable ancestors of the outcome. It excludes immutable variables, evaluates allowed state changes, and ranks plans meeting the threshold by weighted change cost. Downstream variables are free to respond to actions; fixing them at their old values would block causal pathways incorrectly.

For each plan, the displayed baseline forces the same variables to their current values. The action forces them to the proposed values, with the same safe pre-action context. Different action subsets can have different baselines. Costs are normalized state distances with user-selected weights; they are not currency amounts.

**If “No action needed” appears:** the model's current-setting probability already meets the threshold. This may occur even for a record whose observed outcome was unfavorable; predictions are probabilistic. Try a poorer profile or a higher threshold.

**If no feasible plan appears:** the constraints or graph do not permit an action meeting the requested threshold. Lower the threshold or reconsider editable features. Do not conceal this; a correct optimizer must be able to report no solution.

These are conditional intervention scenarios, not identified individual counterfactuals. A full personal counterfactual needs additional structural assumptions and abduction of the individual's unobserved noise.

### Minute 6–8: prove that evaluation is meaningful

Open **Model lab**.

> All three predictive models are tested on the same held-out rows. A majority classifier is a simple sanity baseline; random forest is a stronger predictive comparison. We report the actual result even if another model wins.

Explain:

| Metric | Simple explanation |
|---|---|
| Accuracy | Fraction of test outcomes predicted correctly. |
| Macro F1 | F1 calculated per class and averaged equally; helps expose poor minority-class performance. |
| Log loss | Penalizes wrong or overconfident probabilities; lower is better. |
| Confusion matrix | Counts actual versus predicted classes, including the mistakes. |
| Directed edge precision | Of our proposed arrows, the fraction matching known truth. |
| Directed edge recall | Of known true arrows, the fraction recovered. |
| Graph F1 | Combines directed edge precision and recall. |
| SHD | Structural edits needed to match truth; a reversed edge counts once. Lower is better. |

A good predictive score does not prove a good causal graph. Structural metrics are available only when the synthetic ground truth is known. BIC scores should be compared on the same data and variable schema, not casually across different datasets.

### Minute 8–9: architecture and conclusion

Say:

> React provides the interactive frontend. FastAPI validates requests and hosts the production build. The Python service handles preprocessing, structure learning, Bayesian estimation and held-out evaluation. pgmpy performs probabilistic inference, NetworkX manages graph operations, and scikit-learn supplies the split, metrics and predictive baseline.

Pipeline:

```text
Dataset → Stratified split → Train-only preprocessing
                          → Learn graph → Fit probability tables
                          → Evaluate on held-out records
                          → Explore graph / Intervene / Search actions
```

Closing:

> The contribution is an explainable, integrated decision workflow. Its limitations are synthetic benchmarks, discrete approximations, assumptions about causal direction, and bounded combinatorial search. Our next steps are expert-validated graphs, real datasets and uncertainty analysis.

## 4. Common evaluator questions

**Where is the AI?** Structure learning learns dependencies; Bayesian estimation learns conditional distributions; variable elimination performs reasoning under uncertainty; the planner searches constrained actions. This is probabilistic AI.

**Why Bayesian networks instead of only random forest?** Random forest is a strong predictive baseline. The graph exposes dependencies and supports interventions under explicit causal assumptions. Interpretability and intervention analysis are the main reasons.

**What is a DAG?** A directed acyclic graph: arrows have a direction and cannot form a directed cycle.

**What is a CPD?** A conditional probability distribution describing a variable's probabilities for each configuration of its parents.

**What is Bayesian smoothing?** Adding prior pseudo-counts makes estimates less brittle when some parent/state combinations are rare or absent in training.

**What is the difference between association and intervention?** Conditioning selects observed cases. Intervention changes a generating mechanism by forcing a state and removing incoming arrows.

**Can observational data prove causality?** Not on its own in general. Identification needs assumptions and often expert knowledge or experiments. Equivalent graphs and hidden confounders remain limitations.

**What is the PC algorithm?** A constraint-based approach that uses conditional independence tests to remove and orient edges. Its conclusions depend on test reliability and causal assumptions.

**What is TAN?** Tree-Augmented Naive Bayes adds a tree of feature dependencies to a classifier structure. It is useful for prediction but its arrow directions are not causal evidence by themselves.

**Is the project novel research?** It integrates established methods into an application and improves validation, evaluation and usability. Do not claim a newly invented causal-learning algorithm.

**What did we improve?** A React/FastAPI architecture, animated interactive graph, train-only preprocessing, shared held-out evaluation and baselines, actual state-name handling, d-separation-based adjustment search, correct reversal counting in SHD, bounded scenario search, safe downstream propagation, exports and an embedded demonstration guide.

**Does it use an LLM?** No. No text-generation API or paid AI account is required.

**What are the limitations of the action cost?** Numeric state differences are normalized, so category ordering may be only an approximation. Costs are configurable weights, not real monetary or effort measurements. Real applications need expert-defined actions and feasibility constraints.

**Is it ready for clinical or credit deployment?** It is an academic research application. It has not been externally validated on a real population and does not guarantee outcomes.

**What tests did we perform?** Use `docs/VALIDATION_REPORT.md` for the final verified test counts and coverage. Distinguish automated workflow checks from causal validity.

## 5. Practical troubleshooting

- **Page unavailable:** keep the launcher running, and use port 8000, not the former Streamlit port 8501.
- **First launch takes time:** Python imports and the initial training run happen before the overview is ready.
- **Workspace expired:** the local server stores the eight most recent sessions in memory; build the workspace again.
- **Uploaded CSV rejected:** inspect the error; remove missing values, IDs and extremely rare categories. Use a categorical target.
- **Search too large:** reduce maximum actions or lock more features.
- **Wrong/old numbers:** check domain, algorithm, record count and seed. Rebuild the workspace.
- **Port already in use:** use the existing project instance or stop its launcher before starting another one.
