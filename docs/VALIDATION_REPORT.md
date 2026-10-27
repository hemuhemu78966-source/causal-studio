# Validation report - Causal Studio 2.0

Validated on 26 October 2026 on this Windows computer using Python 3.12, Node.js 24 and a production React build served by FastAPI on localhost:8000.

## Automated results

| Check | Result |
|---|---|
| Python test suite | 43 passed |
| Production React build | Passed |
| Browser workflow checks | 14 passed |
| Browser JavaScript errors during checks | 0 |
| Desktop horizontal overflow | None at 1440px |
| Mobile horizontal overflow | None at 390px |

The Python suite includes the original 18 tests plus 25 new semantic, pipeline and API checks. Library deprecation warnings remain for pgmpy's estimator interfaces; they did not cause test failures. pgmpy is pinned to the tested 1.1.2 release.

Coverage includes all five benchmark domains, Hill-Climbing/PC/TAN/reference modes, train-only transformations, custom string categories, distinct same-size uploads, probability normalization, valid model CPDs, d-separation compatibility, confounder and collider handling, text-valued interventions, invalid-state rejection, deterministic target interventions, mediation propagation, immutable constraints, no-action and no-feasible-plan cases, reversed-edge SHD, HTTP validation and frontend hosting.

Browser checks cover the overview, graph controls and reference toggle, intervention calculation, scenario export, recommendation search with immutable Age, model evaluation display, readable-label search, CSV download, presentation guide, keyboard modal dismissal, custom CSV training, invalid API requests, report export and mobile navigation.

Screenshots were inspected for desktop overview, recommendation layout and mobile overview. These are selected states, not exhaustive inspection of every setting. Browser checks ran in Chromium; other browser engines were not tested.

## Reproducible default result

Configuration: Credit intelligence, Hill-Climbing with BIC, 2,000 synthetic rows, seed 42, 1,600 train / 400 test.

| Model | Accuracy | Macro F1 | Log loss |
|---|---:|---:|---:|
| Bayesian network | 77.00% | 0.7623 | 0.5025 |
| Random forest | 75.25% | 0.7408 | 0.5132 |
| Majority baseline | 59.75% | 0.3740 | 0.6740 |

Savings low to high produced a modeled approval probability change of approximately 30.04% to 85.89% (+55.85 percentage points). This depends on the model's causal interpretation.

The tested action-planner profile sets Savings to Low and Credit Score to Poor, keeps Age immutable and uses a 65% threshold and maximum two changes. The best one-feature scenario changed Savings to High at normalized cost 1.00, with modeled probability about 85.9%. A second changed Credit Score to Excellent, with probability about 78.4%. A two-feature alternative reached about 89.8% at cost 2.00. Each plan uses its own same-variable current-setting comparison.

## Limits of validation

Passing tests establishes the behavior exercised by these checks. It does not establish real-world causal identification, fairness, clinical/credit suitability, population generalization, uncertainty coverage or guaranteed action effectiveness. Built-in benchmarks are synthetic. Uploaded-data graphs require domain validation. The web planner reports intervention scenarios, not identified personal counterfactuals.

Standalone legacy SCM/counterfactual utilities are not web application features. Existing unit tests cover selected SCM examples; this validation does not certify every legacy research utility.

## Evidence and rerun commands

- artifacts/test-results.txt
- artifacts/browser-results.json
- artifacts/desktop-overview.png
- artifacts/desktop-intervention.png
- artifacts/desktop-planner.png
- artifacts/desktop-evaluation.png
- artifacts/mobile-overview.png
- artifacts/mobile-intervention.png

```powershell
.\.venv\Scripts\python.exe -m pytest tests -q
npm --prefix frontend run build
node frontend\e2e.mjs
```

Run browser checks from the project root with the API running. The suite generates its own synthetic custom CSV.
