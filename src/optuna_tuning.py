import optuna
import joblib
import mlflow
import mlflow.sklearn

from xgboost import XGBClassifier
from sklearn.metrics import f1_score


def tune_xgboost_with_optuna(
    X_res,
    y_res,
    X_val,
    y_val,
    baseline_xgb_macro_f1=None
):
    """
    TODO:
    - Run Optuna study (30 trials)
    - Tune XGBoost hyperparameters
    - Optimize macro_f1 on validation set
    - Register best model in MLflow
    - Promote model to production alias
    - Save model using joblib
    """

    # =========================
    # TODO: Configure Optuna + MLflow
    # =========================
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment("PredMaint_Optuna")

    # =========================
    # TODO: Define Optuna objective
    # =========================
    def objective(trial):

        params = {
            "n_estimators": trial.suggest_int(
                "n_estimators",
                100,
                500
            ),

            "max_depth": trial.suggest_int(
                "max_depth",
                3,
                10
            ),

            "learning_rate": trial.suggest_float(
                "learning_rate",
                0.01,
                0.3,
                log=True
            ),

            "min_child_weight": trial.suggest_float(
                "min_child_weight",
                1.0,
                10.0
            ),

            "subsample": trial.suggest_float(
                "subsample",
                0.6,
                1.0
            ),

            "colsample_bytree": trial.suggest_float(
                "colsample_bytree",
                0.6,
                1.0
            ),

            "gamma": trial.suggest_float(
                "gamma",
                0.0,
                2.0
            ),

            "reg_alpha": trial.suggest_float(
                "reg_alpha",
                1e-8,
                1.0,
                log=True
            ),

            "reg_lambda": trial.suggest_float(
                "reg_lambda",
                1e-8,
                5.0,
                log=True
            ),

            "random_state": 42,
            "eval_metric": "mlogloss",
            "verbosity": 0,
            "n_jobs": -1
        }

        # =========================
        # TODO: Train XGBoost model
        # =========================
        model = XGBClassifier(**params)

        model.fit(X_res, y_res)

        # =========================
        # TODO: Predict validation set
        # =========================
        y_pred = model.predict(X_val)

        # =========================
        # TODO: Compute macro F1
        # =========================
        macro_f1 = f1_score(
            y_val,
            y_pred,
            average="macro"
        )

        return macro_f1

    # =========================
    # TODO: Create Optuna study
    # =========================
    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=42)
    )

    # =========================
    # TODO: Run optimization
    # =========================
    print("\n========== RUNNING OPTUNA ==========")

    study.optimize(
        objective,
        n_trials=30
    )

    print("\n========== OPTUNA RESULTS ==========")
    print("Best Params:")
    print(study.best_params)

    print("\nBest Macro F1:")
    print(study.best_value)

    # =========================
    # TODO: Train final best model
    # =========================
    best_params = study.best_params

    best_params.update({
        "random_state": 42,
        "eval_metric": "mlogloss",
        "verbosity": 0,
        "n_jobs": -1
    })

    best_model = XGBClassifier(**best_params)

    best_model.fit(X_res, y_res)

    # =========================
    # TODO: Final evaluation
    # =========================
    y_pred_best = best_model.predict(X_val)

    best_macro_f1 = f1_score(
        y_val,
        y_pred_best,
        average="macro"
    )

    print("\n========== FINAL MODEL PERFORMANCE ==========")

    print(f"Baseline XGBoost Macro F1 : {baseline_xgb_macro_f1}")
    print(f"Tuned XGBoost Macro F1    : {best_macro_f1}")

    if baseline_xgb_macro_f1 is not None:
        improvement = best_macro_f1 - baseline_xgb_macro_f1

        print(f"Improvement               : {improvement}")

    # =========================
    # TODO: Log model to MLflow
    # =========================
    with mlflow.start_run(run_name="Optuna_Best_XGBoost"):

        mlflow.log_params(best_params)

        mlflow.log_metric(
            "macro_f1",
            best_macro_f1
        )

        model_info = mlflow.sklearn.log_model(
            sk_model=best_model,
            artifact_path="model",
            registered_model_name="PredMaint_XGBoost",
            input_example=X_val.iloc[:5]
        )

        print("\nModel logged to MLflow.")
        print("Registered model name: PredMaint_XGBoost")

    # =========================
    # TODO: Promote model alias
    # =========================
    client = mlflow.tracking.MlflowClient()

    latest_version = client.get_latest_versions(
        "PredMaint_XGBoost"
    )[-1].version

    client.set_registered_model_alias(
        name="PredMaint_XGBoost",
        alias="production",
        version=latest_version
    )

    print(
        f"\nPromoted PredMaint_XGBoost "
        f"version {latest_version} "
        f"to alias 'production'"
    )

    # =========================
    # TODO: Save model locally
    # =========================
    joblib.dump(
        best_model,
        "best_xgboost_model.pkl"
    )

    print("\nSaved model: best_xgboost_model.pkl")

    return (
        best_model,
        best_macro_f1,
        best_params
    )


def main():
    """
    Main function for standalone execution
    """

    print("This module is intended to be imported into Jupyter Notebook.")
    print("Run tune_xgboost_with_optuna() after model selection stage.")


if __name__ == "__main__":
    main()