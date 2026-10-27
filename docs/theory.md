# Causal Bayesian Networks - Theoretical Background

## Introduction

Causal Bayesian Networks (CBNs) extend traditional Bayesian networks by explicitly modeling causal relationships between variables. This enables reasoning about interventions and counterfactuals, which are essential for decision-making.

## Fundamental Concepts

### 1. Causality vs. Correlation

- **Correlation**: Statistical association between variables (X and Y co-occur)
- **Causation**: X directly influences Y (manipulating X changes Y)

Traditional statistics focuses on correlation. Causal inference answers:
- **Prediction**: What will happen? P(Y|X)
- **Intervention**: What if we do X? P(Y|do(X))
- **Counterfactual**: What if X had been different? P(Y_x|X',Y')

### 2. Directed Acyclic Graphs (DAGs)

A causal DAG represents:
- **Nodes**: Variables
- **Edges**: Direct causal relationships
- **Direction**: Arrow from cause to effect

**Properties**:
- Acyclic: No variable can be its own ancestor
- Markov condition: Each variable is independent of its non-descendants given its parents

### 3. Structural Causal Models (SCMs)

An SCM consists of:
- **Endogenous variables** (V): Variables modeled in the system
- **Exogenous variables** (U): External factors/noise
- **Structural equations**: f_i : Pa_i × U_i → V_i

Example:
```
U_X ~ N(0,1)
U_Y ~ N(0,1)
X = U_X
Y = 2X + U_Y
```

## Do-Calculus and Interventions

### Intervention Operator: do(X=x)

The do-operator represents setting X to value x, breaking all incoming causal links.

**Graphical operation**: Remove all edges into X

**Difference from conditioning**:
- P(Y|X=x): Observational (passive observation)
- P(Y|do(X=x)): Interventional (active manipulation)

### Rules of Do-Calculus

Pearl's three rules allow computing interventional distributions from observational data:

1. **Rule 1** (Insertion/deletion of observations)
2. **Rule 2** (Action/observation exchange)
3. **Rule 3** (Insertion/deletion of actions)

### Backdoor Criterion

To identify P(Y|do(X)), we need to block all "backdoor paths" from X to Y.

**Backdoor paths**: Paths with an arrow into X

**Adjustment formula**: If Z satisfies the backdoor criterion:
```
P(Y|do(X=x)) = Σ_z P(Y|X=x,Z=z) P(Z=z)
```

## Counterfactual Reasoning

### Three-Step Process

1. **Abduction**: Update belief about exogenous variables given observations
   - Infer U from observed (X,Y)

2. **Action**: Modify the model to reflect the intervention
   - Replace equation for X with X = x

3. **Prediction**: Use modified model to compute outcome
   - Compute Y from modified equations

### Example

**Question**: Patient had disease (D=1), no treatment (T=0), and died (R=0). Would they have survived with treatment?

**SCM**:
```
D = U_D
T = D + U_T
R = 0.9T - 0.6D + U_R
```

**Abduction**: Given (D=1, T=0, R=0), infer (U_D, U_T, U_R)

**Action**: Set T=1

**Prediction**: Compute R with new T and inferred noise terms

### Probability of Causation

- **PN (Necessity)**: P(Y_0=0 | X=1, Y=1)
  - "X was necessary for Y"
  
- **PS (Sufficiency)**: P(Y_1=1 | X=0, Y=0)
  - "X would have been sufficient for Y"

- **PNS (Necessity and Sufficiency)**: P(Y_1=1, Y_0=0)
  - "X was both necessary and sufficient"

## Treatment Effects

### Average Treatment Effect (ATE)

```
ATE = E[Y|do(T=1)] - E[Y|do(T=0)]
```

Average effect across entire population.

### Conditional Average Treatment Effect (CATE)

```
CATE(x) = E[Y|do(T=1),X=x] - E[Y|do(T=0),X=x]
```

Effect for subgroup defined by X=x (heterogeneous effects).

### Individual Treatment Effect (ITE)

```
ITE_i = Y_i(1) - Y_i(0)
```

Effect for individual i (counterfactual, generally unobservable).

## Applications

### 1. Medicine
- Treatment effect estimation
- Personalized medicine
- Clinical decision support
- Adverse event analysis

### 2. Economics and Policy
- Policy evaluation
- Impact assessment
- Resource allocation
- Program effectiveness

### 3. Business
- Marketing campaign analysis
- Pricing strategies
- A/B test interpretation
- Customer lifetime value

### 4. Fairness and Discrimination
- Detecting discrimination
- Fair decision-making
- Bias correction

## Identifiability

**Key question**: Can we compute P(Y|do(X)) from observational data?

**Sufficient conditions**:
- Backdoor criterion satisfied
- Front-door criterion satisfied
- Do-calculus derivation exists

**Non-identifiable cases**:
- Unmeasured confounding
- Selection bias
- Feedback loops

**Sensitivity analysis**: Bound causal effects under assumptions about unobserved confounders.

## Further Reading

1. Pearl, J. (2009). *Causality: Models, Reasoning, and Inference*. Cambridge University Press.

2. Pearl, J., & Mackenzie, D. (2018). *The Book of Why*. Basic Books.

3. Peters, J., Janzing, D., & Schölkopf, B. (2017). *Elements of Causal Inference*. MIT Press.

4. Hernán, M. A., & Robins, J. M. (2020). *Causal Inference: What If*. Chapman & Hall/CRC.

5. Imbens, G. W., & Rubin, D. B. (2015). *Causal Inference for Statistics, Social, and Biomedical Sciences*. Cambridge University Press.
