# Predictive Maintenance Classification — MLOps

## Project Overview
This project is an end-to-end **MLOps pipeline** for predictive maintenance in an Industrial IoT / Manufacturing domain.  
The objective is to build a multi-class classification model that predicts machine failure types using sensor readings.

The solution covers:
- Data validation using **Pandera**
- Model training + experiment tracking using **MLflow**
- Hyperparameter tuning using **Optuna**
- Drift monitoring using **Evidently**
- Explainability using **SHAP**
- Automated pipeline execution using `run_pipeline.py`
- CI/CD ready structure (GitHub Actions supported)

---

## Failure Classes

| Code | Name | Description |
|------|------|-------------|
| 0 | No Failure | Machine operating normally |
| 1 | TWF | Tool Wear Failure |
| 2 | HDF | Heat Dissipation Failure |
| 3 | PWF | Power Failure |
| 4 | OSF | Overstrain Failure |

---

## Dataset Files

All datasets are stored inside the `data/` folder:

- `data/train.csv`   → baseline training dataset (6,993 rows)
- `data/current.csv` → current production batch (1,499 rows)
- `data/stress.csv`  → stress/heavy-load batch (1,499 rows)

Each dataset contains:
- 6 sensor features + categorical machine type
- target column: `Failure_Type`

---

## Project Structure

```bash
.
├── data/
│   ├── train.csv
│   ├── current.csv
│   └── stress.csv
│
├── notebooks/
│   └── analysis.ipynb
│
├── src/
│   ├── load_data.py
│   ├── data_validation.py
│   ├── eda.py
│   ├── feature_engineering.py
│   ├── preprocessing.py
│   ├── model_training.py
│   ├── optuna_tuning.py
│   ├── drift_detection_current.py
│   ├── drift_detection_stress.py
│   └── shap_analysis.py
│
├── mlruns/                # MLflow runs directory
├── mlflow.db              # MLflow sqlite database
├── run_pipeline.py        # Pipeline runner script
├── requirements.txt
├── best_xgboost_model.pkl
├── drift_current.html
├── drift_stress.html
├── shap_per_class.png
└── README.md
