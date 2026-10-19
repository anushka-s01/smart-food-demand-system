# Audit and handover

## Files received

- `app.py`: four-page Streamlit interface and integration layer.
- `database.py`: SQLite initialization, insertion, and history retrieval.
- `model.py`: pandas-based fallback predictor.
- `data_processing.py`: dataset schema validation and feature preparation.
- `food_logic.py`: fixed 5% preparation buffer and surplus-risk label.
- `dataset.csv`: 300 complete, duplicate-free synthetic records.
- `model.pkl`: trusted project artifact containing a scikit-learn pipeline with
  one-hot encoding and a 200-tree `RandomForestRegressor`.
- `requirements.txt`: backend dependencies only.
- `README.md`: backend instructions that did not describe the complete app.
- `smartmess.db`: two local demonstration rows; deliberately excluded from the
  finished project because it is generated state.

The handover also referred to `generate_data.py` and `test_backend.py`, but
neither file was included in the received set.

The named GitHub repository, `https://github.com/znb-kn/smart-mess`, was
publicly reachable during the audit. Its current root contained `app.py`,
`data_processing.py`, `database.py`, `dataset.csv`, `food_logic.py`, `model.py`,
and `requirements.txt`. It did not contain the supplied `model.pkl`, a README,
a `.gitignore`, or tests, and its requirements did not include Streamlit or
Plotly. The repository itself was inspected read-only and was not modified.

## Audit findings

The four Streamlit pages, database save flow, history, and analytics were
substantially implemented. Database queries were parameterized, and empty
states existed.

The main functional defect was in `model.py`: it searched for several possible
target names but omitted the dataset's actual target, `Actual_Attendance`.
Consequently, it returned `previous_attendance` for every input. The frontend's
menu choices also differed from the categories used to train `model.pkl`.

The interface described every result as an ML prediction even when the
fallback was active. It called the planning buffer “expected wastage,” which can
be confused with measured waste. The session-state flow could preserve an old
result while the user edited inputs, and there was no protection against
repeated saves within the same session.

`requirements.txt` omitted Streamlit and Plotly. The database path depended on
the terminal's current directory. The README did not explain deployment or
SQLite persistence. There was no `.gitignore` in the received set.

## Changes made

- Aligned the dropdown values, dataset validation, and model categories.
- Restored the supplied Random Forest pipeline as the primary predictor.
- Added a historical-average fallback that is clearly disclosed in the app.
- Added reproducible model training to `model.py`.
- Replaced user-facing “expected wastage” wording with “potential surplus.”
- Used a Streamlit form so a prediction corresponds to one submitted input set.
- Prevented repeated saving of the same result during a session.
- Made SQLite initialization repeatable and its path independent of the shell's
  working directory.
- Added analytics filters and a small-sample explanation.
- Added complete runtime dependencies, repository exclusions, tests, local
  setup instructions, and deployment guidance.

## Current limitations

- All 300 dataset rows are synthetic; model performance does not prove
  real-world accuracy.
- The fixed 5% buffer normally produces a Low risk label, so the label has
  limited analytical value until actual waste data supports a better rule.
- The supplied model depends on compatible scikit-learn and joblib versions.
  The fallback keeps the demonstration usable if those libraries cannot load.
- SQLite on Streamlit Community Cloud is not durable across restarts or
  redeployments.
- Public deployment still requires the owner's GitHub and Streamlit accounts
  and must be verified using the resulting URL.

## Verification completed on 19 October 2026

- Dataset validation: 300 rows, no missing required values, and no duplicate
  rows.
- Reproducible training: held-out MAE was 26.08 students on the synthetic
  dataset. This value is a development metric, not evidence of campus accuracy.
- Automated core checks: 9 of 9 passed.
- Streamlit interaction checks: empty dashboard, prediction, single save,
  duplicate-save protection, populated dashboard, history, and all four
  analytics charts passed.
- Runtime smoke check: the Streamlit server started and its health endpoint
  returned `ok`.

Public deployment was not performed because it requires access to the project
owner's GitHub and Streamlit accounts. Completion should only be claimed after
the public URL is opened and tested in a separate browser session.

## Demonstration checklist

1. Open the Dashboard and explain the empty state.
2. Go to Demand Prediction and enter one meal scenario.
3. Explain predicted attendance, recommended portions, and potential surplus.
4. Save the result once.
5. Return to the Dashboard and show the latest metrics.
6. Open History and show the saved row.
7. Open Analytics and show how the charts use saved database records.
8. Point out the prediction method in the sidebar.
9. State that the data is synthetic and that real mess data is future work.

## Brief explanation for an evaluator

“The application takes the day, meal, menu, holiday status, examination status,
and previous attendance as inputs. A Random Forest pipeline estimates expected
attendance. The food logic adds a 5% planning buffer to recommend portions, and
the difference is shown as potential surplus rather than actual waste. When a
prediction is saved, SQLite stores it for the dashboard, history, and analytics.
The current dataset is synthetic, so the project demonstrates the complete
workflow; real campus data is needed for operational validation.”
