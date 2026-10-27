# Project Setup and Usage Guide

## Installation

### 1. Clone the Repository

```bash
git clone <repository-url>
cd causal-bayesian-networks
```

### 2. Create Virtual Environment

**On Windows:**
```powershell
python -m venv venv
.\venv\Scripts\activate
```

**On Linux/Mac:**
```bash
python -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Verify Installation

```bash
python -c "import src; print('Installation successful!')"
```

## Running Examples

### Medical Treatment Example

```bash
python examples/medical_treatment.py
```

This demonstrates:
- Building a causal network
- Computing average treatment effects
- Performing do-calculus interventions

## Running Tests

```bash
# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=src --cov-report=html

# Run specific test file
pytest tests/test_models.py -v
```

## Using Jupyter Notebooks

```bash
# Start Jupyter
jupyter notebook

# Navigate to notebooks/ folder
# Open 01_introduction.ipynb
```

## Project Structure

```
causal-bayesian-networks/
├── src/                    # Source code
│   ├── models/            # Network and SCM implementations
│   ├── inference/         # Inference algorithms
│   ├── utils/             # Helper functions
│   └── visualization/     # Plotting utilities
├── data/                  # Datasets
├── notebooks/             # Jupyter notebooks
├── tests/                 # Unit tests
├── docs/                  # Documentation
├── examples/              # Example scripts
├── results/               # Output directory
└── requirements.txt       # Python dependencies
```

## Basic Usage

### Creating a Causal Network

```python
from src.models.bayesian_network import CausalBayesianNetwork
from pgmpy.factors.discrete import TabularCPD

# Initialize network
cbn = CausalBayesianNetwork()

# Add variables
cbn.add_nodes(['X', 'Y', 'Z'])

# Define causal structure
cbn.add_edges([('X', 'Y'), ('Y', 'Z')])

# Define conditional probability distributions
cpd_x = TabularCPD('X', 2, [[0.6], [0.4]])
cbn.set_cpd('X', cpd_x)

# Build model
cbn.build_model()
```

### Performing Interventions

```python
from src.inference.intervention import InterventionAnalysis

# Initialize intervention analysis
intervention = InterventionAnalysis(cbn)

# Compute P(Y | do(X=1))
result = intervention.do_intervention({'X': 1}, 'Y')

# Compute average treatment effect
ate = intervention.average_treatment_effect('X', 'Y')
print(f"ATE: {ate}")
```

### Counterfactual Queries

```python
from src.models.structural_causal_model import StructuralCausalModel
from src.inference.counterfactual import CounterfactualInference

# Create SCM
scm = StructuralCausalModel(cbn)

# Define structural equations
scm.add_structural_equation(
    'Y',
    lambda parents, noise: parents['X'] + noise
)

# Perform counterfactual inference
cf_inference = CounterfactualInference(cbn, scm)
result = cf_inference.query_counterfactual(
    observed={'X': 0, 'Y': 1},
    intervention={'X': 1},
    target='Y'
)
```

### Visualization

```python
from src.visualization.plotting import plot_causal_graph

# Visualize network structure
plot_causal_graph(cbn, save_path='results/network.png')
```

## Common Tasks

### Adding New Datasets

Place data files in `data/` directory:
```
data/
├── raw/           # Original datasets
└── processed/     # Cleaned datasets
```

### Saving Results

```python
from src.utils.helpers import save_results

results = {
    'ate': 0.25,
    'confidence_interval': (0.15, 0.35)
}

save_results(results, 'results/analysis_results.json')
```

### Creating Custom Models

Extend the base classes:

```python
from src.models.bayesian_network import CausalBayesianNetwork

class CustomNetwork(CausalBayesianNetwork):
    def __init__(self):
        super().__init__()
        # Add custom initialization
    
    def custom_method(self):
        # Add custom functionality
        pass
```

## Troubleshooting

### Import Errors

If you get import errors, ensure you're in the project root and the virtual environment is activated:

```bash
cd causal-bayesian-networks
source venv/bin/activate  # or .\venv\Scripts\activate on Windows
```

### Missing Dependencies

```bash
pip install --upgrade -r requirements.txt
```

### Test Failures

Check that all dependencies are correctly installed and you're using Python 3.8+:

```bash
python --version
pip list
```

## Next Steps

1. Read `docs/theory.md` for theoretical background
2. Work through `notebooks/01_introduction.ipynb`
3. Explore `examples/` directory
4. Build your own causal models
5. Contribute improvements via pull requests

## Getting Help

- Check documentation in `docs/`
- Review example code in `examples/`
- Look at test cases in `tests/`
- Open an issue on GitHub

## Citation

If you use this code in your research, please cite:

```bibtex
@software{causal_bayesian_networks,
  title = {Causal Bayesian Networks for Counterfactual Decision Analysis},
  author = {Your Name},
  year = {2026},
  url = {https://github.com/yourusername/causal-bayesian-networks}
}
```
