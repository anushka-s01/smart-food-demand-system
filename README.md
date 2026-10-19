# Smart Food Demand Prediction and Waste Reduction System

A Streamlit application that predicts campus meal attendance, recommends a
preparation quantity with a 5% planning buffer, saves planning records in
SQLite, and presents a dashboard, history, and analytics.

## What the prediction means

The included Random Forest pipeline was trained on 300 **synthetic** records.
It demonstrates the complete machine-learning workflow but does not establish
real-world accuracy. The application labels calculated excess portions as
**potential surplus**. Actual food waste must be measured after a meal.

If the trained model cannot be loaded, the app continues with a documented
historical-average fallback and displays that method in the sidebar.

## Run locally in VS Code on Windows

Use Python 3.12 for the most predictable package compatibility.

Open this folder in VS Code, then run these commands in a PowerShell terminal:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m unittest -v test_project.py
python -m streamlit run app.py
```

If PowerShell blocks activation, the same commands can be run without
activation:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m unittest -v test_project.py
.\.venv\Scripts\python.exe -m streamlit run app.py
```

Open the local URL shown in the terminal, usually `http://localhost:8501`.

## Demonstration history

On a fresh installation or Streamlit Community Cloud deployment, the app
automatically creates 300 realistic sample records when the prediction database
starts. The startup check is transaction-safe, repairs duplicate demo rows, and
never removes user predictions. Every generated row is labelled `Synthetic demo`
rather than measured campus activity.

To recreate the demonstration history manually during local development, run:

```powershell
python seed_demo_history.py --count 300 --replace
```

The `--replace` option removes only older rows labelled `Synthetic demo`; it
preserves predictions saved by the user.

## Retrain the model

`model.py` contains the reproducible preprocessing and training pipeline. Run:

```powershell
python model.py
```

This validates `dataset.csv`, trains the model, prints the held-out mean
absolute error for this synthetic dataset, and replaces `model.pkl`.

## Files required for GitHub and deployment

- `app.py`
- `database.py`
- `model.py`
- `data_processing.py`
- `food_logic.py`
- `dataset.csv`
- `model.pkl`
- `requirements.txt`
- `seed_demo_history.py`
- `.streamlit/config.toml`
- `.gitignore`

The test and documentation files should also be committed. Do not commit the
generated `food_demand.db` file.

## Deploy on Streamlit Community Cloud

1. Create a GitHub repository and add the project files.
2. Confirm that `food_demand.db`, `.venv`, and `__pycache__` are excluded.
3. In Streamlit Community Cloud, select the repository and `main` branch.
4. Set the entry point to `app.py`.
5. Select Python 3.12 in Advanced settings when that option is available.
6. Deploy, then test prediction, saving, dashboard, analytics, and history from
   a separate browser session.

SQLite data on Streamlit Community Cloud may disappear when the app restarts or
is redeployed because the local filesystem is not permanent. If this happens,
the app recreates the 300 labelled synthetic demo records automatically. User
predictions saved only in that temporary database may still disappear. For a
classroom demo this is acceptable. Long-term shared history should use a
persistent hosted database in a later version.
