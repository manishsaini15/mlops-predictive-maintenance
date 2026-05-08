import shap
import joblib
import numpy as np
import matplotlib.pyplot as plt

from load_data import load_datasets
from feature_engineering import engineer_features
from preprocessing import encode_type_column, FEATURES



FAILURE_CLASSES = {
    1: "TWF",
    2: "HDF",
    3: "PWF",
    4: "OSF"
}


def run_shap_analysis(
    train,
    features,
    model_path="best_xgboost_model.pkl",
    save_path="shap_per_class.png"
):
    """
    TODO: Compute SHAP values using TreeExplainer
    TODO: Plot 4-subplot bar chart (one per failure class)
    TODO: Save shap_per_class.png
    TODO: Print top driver per class
    """

    print("\nRunning SHAP Analysis...\n")

    # Load best model
    best_model = joblib.load(model_path)

    X_shap = train[features]

    # TreeExplainer
    explainer = shap.TreeExplainer(best_model)

    # SHAP values (multiclass)
    shap_values = explainer.shap_values(X_shap)

    # Convert to numpy array of shape:
    # (n_samples, n_features, n_classes)
    shap_array = np.array(shap_values)
    shap_array = np.transpose(shap_array, (1, 2, 0))

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()

    top_drivers = {}

    for i, (cls, name) in enumerate(FAILURE_CLASSES.items()):

        # Mean absolute SHAP values for class
        mean_abs_shap = np.abs(shap_array[:, :, cls]).mean(axis=0)

        # Sort features
        sorted_idx = np.argsort(mean_abs_shap)[::-1]

        sorted_features = np.array(features)[sorted_idx]
        sorted_vals = mean_abs_shap[sorted_idx]

        # Store top driver
        top_drivers[name] = sorted_features[0]

        # Plot
        axes[i].barh(
            sorted_features[::-1],
            sorted_vals[::-1]
        )

        axes[i].set_title(f"Mean |SHAP| - {name}")
        axes[i].set_xlabel("Mean |SHAP value|")

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()

    print(f"Saved SHAP plot as {save_path}")

    # Print top driver per class
    print("\nTop driver per failure class:")

    for cls_name, feature in top_drivers.items():
        print(f"{cls_name}: {feature}")

    return shap_array, top_drivers


def main():

    print("\nLoading datasets for SHAP analysis...\n")

    train, current, stress = load_datasets()

    # Feature Engineering
    train = engineer_features(train)
    current = engineer_features(current)
    stress = engineer_features(stress)

    # Encoding
    train, current, stress, le = encode_type_column(
        train,
        current,
        stress
    )

    # Run SHAP
    run_shap_analysis(
        train=train,
        features=FEATURES
    )

    print("\nSHAP Analysis Completed!\n")


if __name__ == "__main__":
    main()