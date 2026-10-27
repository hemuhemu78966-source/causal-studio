# Causal Studio 2.0

An explainable AI research application for Bayesian structure learning, intervention analysis and cost-aware action scenarios.

**Frontend:** React 19 + Vite, custom responsive CSS, Lucide icons and interactive animated SVG graphs. **Backend:** FastAPI, pgmpy, NetworkX, pandas and scikit-learn. The current app does not use Streamlit.

## Run on this computer

Double-click **Launch Causal Studio.bat**, or run from this folder:

```powershell
.\.venv\Scripts\python.exe app.py
```

Open http://127.0.0.1:8000. Keep the launcher window running. Stop with Ctrl+C. If the port is already occupied by this project, use the existing instance rather than launching another copy.

## First-time setup on another computer

Requires Python 3.12 and Node.js 20.19+ or 22.12+.

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
npm --prefix frontend ci
npm --prefix frontend run build
.\.venv\Scripts\python.exe app.py
```

The production build is served by FastAPI from the same origin. Fonts and icons are bundled locally; no hosted AI API, external graph library CDN or internet connection is required after setup.

For frontend development, run the backend separately and then `npm --prefix frontend run dev`. Vite proxies `/api` to port 8000.

## Research workflows

- **Overview:** genuine model statistics, evaluation snapshot and workflow shortcuts.
- **Causal graph:** drag, inspect with mouse or keyboard, zoom and reset nodes; compare learned and known synthetic structures.
- **Interventions:** compare model-conditioned observation with explicit graph-surgery interventions; select the desired outcome and inspect probability changes and adjustment sets.
- **Action planner:** choose a profile, protect immutable features, set positive action costs and search up to three simultaneous changes.
- **Model lab:** held-out accuracy, macro F1, log loss, confusion matrix and structural metrics. Compare Bayesian network, random forest and majority baseline.
- **Dataset:** state labels, distributions, record search, training transformations and downloadable raw CSV.

Choose **Configure model** to change domain, algorithm, sample size, random seed or upload a CSV. Changing settings takes effect when you choose **Build workspace**.

## Data and methodology

Five domains are synthetic: credit, clinical outcomes, retention, student success and employee promotion. Their generating graphs provide known structural ground truth. A reference-graph mode uses that known graph explicitly; it is labeled and is not a learned result. TAN is a predictive graph structure and should not be treated as independently established causal directions.

A stratified 80/20 split is created first. Category vocabularies and continuous-feature quantile boundaries are fitted only on training rows. All graph discovery and CPD fitting use training data. Bayesian, random forest and majority evaluations share the same test rows. BDeu smoothing limits zero-probability estimates.

Uploaded CSVs require 100–10,000 rows, 2–12 columns, no missing/infinite entries, and a categorical target with 2–8 states and at least five rows per state. Numeric features with more than six unique values are divided into three training-quantile bands. Other columns allow up to eight states. Unseen test-only categories produce an explanation rather than silent replacement. Remove identifiers and post-outcome features before training. Very complex graphs and searches above 3,000 candidate scenarios are rejected with guidance.

Sessions are kept in local process memory (most recent eight workspaces). Restarting the server clears them. Downloads preserve your result snapshots; uploads are not written to permanent storage by the API. Exported reports include the first 100 encoded records, so treat them as dataset-containing artifacts.

## Causal interpretation

`P(Y | X=x)` describes observation; `P(Y | do(X=x))` describes an intervention under the assumed causal graph. The latter removes incoming edges into X and assigns a deterministic state. Variable elimination recomputes probabilities.

Action recommendations are **conditional intervention scenarios**. Downstream mediators are allowed to change; they are not held at factual values after an action. Each plan is compared with the same proposed variables forced to their current values using the same pre-action context. Plans have comparable cost units but can use different action subsets and baselines. A no-change plan means current action settings already meet the model threshold. This is not proof of an individual counterfactual, confidence interval, guaranteed improvement or real-world efficacy.

Learned causal interpretations require strong assumptions (including no relevant hidden confounding, suitable measurements, and faithful independence patterns). Predictive accuracy does not establish causality. These limitations are visible in the app.

## Tests

```powershell
.\.venv\Scripts\python.exe -m pytest tests -q
node frontend\e2e.mjs
```

For browser tests, start the API first. Install the test browser once using `npx --prefix frontend playwright install chromium`. Run browser tests from this project directory. Results and screenshots are written to `artifacts/`.

## Source layout

```text
app.py                         React/API launcher
backend/main.py                Validated HTTP API + production static hosting
backend/service.py             Preprocessing, train/test pipeline, evaluation and sessions
frontend/src/App.jsx           React workflows and graph interactions
frontend/src/styles.css        Design system, animation and responsive layouts
src/discovery/                 Structure learning and structural evaluation
src/models/                    Bayesian networks and separate SCM examples
src/inference/                 Graph surgery, inference and adjustment sets
src/recourse/                  Bounded cost-aware scenario search
tests/                         Existing tests and new semantic/API regressions
docs/DEMONSTRATION_GUIDE.md     Full evaluator walkthrough
```

The original Streamlit interface is retained as `app_streamlit_legacy.py` for reference only. The older PowerPoint and theory files predate version 2.0; use the new demonstration guide for current features and claims. The standalone SCM examples remain separate from the web action planner.

## Credits

Built on established algorithms in pgmpy, NetworkX and scikit-learn. Original upstream code retains its MIT license. Describe your contribution accurately: the integrated application, interface, validation, evaluation pipeline and reasoning fixes; do not claim to have invented Bayesian networks or the PC algorithm.
