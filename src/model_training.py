import pandas as pd
import mlflow
import mlflow.sklearn

from sklearn.metrics import f1_score, accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from xgboost import XGBClassifier
from lightgbm import LGBMClassifier


CLASS_LIST = [0, 1, 2, 3, 4]


def get_models():
    """
    Returns dictionary of models to train
    """

    return {
        "LogisticRegression": Pipeline([
            ("scaler", StandardScaler()),
            ("lr", LogisticRegression(
                max_iter=2000,
                random_state=42,
                class_weight="balanced"
            ))
        ]),

        "RandomForest": RandomForestClassifier(
            n_estimators=100,
            random_state=42,
            class_weight="balanced"
        ),

        "XGBoost": XGBClassifier(
            n_estimators=100,
            random_state=42,
            eval_metric="mlogloss",
            verbosity=0
        ),

        "LightGBM": LGBMClassifier(
            n_estimators=100,
            random_state=42,
            class_weight="balanced",
            verbose=-1
        )
    }


def train_and_log_models(X_train_smote, y_train_smote, X_val, y_val):
    """
    TODO:
    - Start MLflow experiment
    - Train all models
    - Evaluate models
    - Log metrics and models to MLflow
    - Return comparison dataframe and best model name
    """

    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment("PredMaint_ModelSelection")

    models_to_run = get_models()

    results = {}

    X_res, y_res = X_train_smote, y_train_smote

    for model_name, model in models_to_run.items():

        print(f"\n========== Training {model_name} ==========")

        with mlflow.start_run(run_name=model_name):

            # =========================
            # TODO: Train model
            # =========================
            model.fit(X_res, y_res)

            # =========================
            # TODO: Predict on validation set
            # =========================
            y_pred = model.predict(X_val)

            # =========================
            # TODO: Compute metrics
            # =========================
            macro_f1 = f1_score(y_val, y_pred, average="macro")
            weighted_f1 = f1_score(y_val, y_pred, average="weighted")
            accuracy = accuracy_score(y_val, y_pred)

            per_class_f1 = f1_score(
                y_val,
                y_pred,
                average=None,
                labels=CLASS_LIST
            )

            # =========================
            # TODO: Log params
            # =========================
            mlflow.log_param("model", model_name)

            # =========================
            # TODO: Log metrics
            # =========================
            mlflow.log_metric("macro_f1", macro_f1)
            mlflow.log_metric("weighted_f1", weighted_f1)
            mlflow.log_metric("accuracy", accuracy)

            for cls, score in zip(CLASS_LIST, per_class_f1):
                mlflow.log_metric(f"f1_class_{cls}", score)

            # =========================
            # TODO: Log model artifact
            # =========================
            mlflow.sklearn.log_model(
                sk_model=model,
                artifact_path="model",
                input_example=X_val.iloc[:5]
            )

            # =========================
            # TODO: Store results
            # =========================
            results[model_name] = {
                "macro_f1": macro_f1,
                "weighted_f1": weighted_f1,
                "accuracy": accuracy,
            }

            for cls, score in zip(CLASS_LIST, per_class_f1):
                results[model_name][f"f1_class_{cls}"] = score

            print(f"{model_name} completed.")
            print(f"Macro F1    : {macro_f1:.4f}")
            print(f"Weighted F1 : {weighted_f1:.4f}")
            print(f"Accuracy    : {accuracy:.4f}")

    # =========================
    # TODO: Create comparison table
    # =========================
    results_df = pd.DataFrame(results).T.sort_values(
        by="macro_f1",
        ascending=False
    )

    # =========================
    # TODO: Identify best model
    # =========================
    best_model_name = results_df.index[0]

    print("\n========== MODEL COMPARISON ==========")
    print(results_df)

    print(f"\nBest model by Macro F1: {best_model_name}")

    return results_df, best_model_name


def main():
    """
    Main function for standalone execution
    """

    print("This module is intended to be imported into Jupyter Notebook.")
    print("Use train_and_log_models() after preprocessing + SMOTE.")


if __name__ == "__main__":
    main()