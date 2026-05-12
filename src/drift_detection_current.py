import pandas as pd

from evidently.legacy.report import Report
from evidently.legacy.metric_preset import DataDriftPreset


FEAT_COLS = [
    "Air temperature",
    "Process temperature",
    "Rotational speed",
    "Torque",
    "Tool wear"
]


def run_current_drift_report(
    train,
    current,
    save_path="drift_current.html"
):
    """
    TODO: Run Evidently on current batch,
    save drift_current.html,
    print summary
    """

    drift_report_current = Report(
        metrics=[DataDriftPreset()]
    )

    drift_report_current.run(
        reference_data=train[FEAT_COLS],
        current_data=current[FEAT_COLS]
    )

    # TODO: Save HTML report
    drift_report_current.save_html(save_path)

    print(f"Saved drift report: {save_path}")

    # TODO: Extract summary
    summary_current = drift_report_current.as_dict()

    drift_detected = summary_current["metrics"][0]["result"]["dataset_drift"]

    n_drifted = summary_current["metrics"][0]["result"]["number_of_drifted_columns"]

    total_cols = summary_current["metrics"][0]["result"]["number_of_columns"]

    print("\nDrift Summary (Current Batch):")
    print("Dataset drift detected?:", drift_detected)
    print(f"Drifted features: {n_drifted} / {total_cols}")

    return drift_report_current, summary_current


def main():

    print("\nRunning Current Batch Drift Detection...\n")

    # TODO: Load datasets
    train = pd.read_csv("data/train.csv")
    current = pd.read_csv("data/current.csv")

    # TODO: Run drift report
    run_current_drift_report(
        train=train,
        current=current,
        save_path="drift_current.html"
    )

    print("\nCurrent Batch Drift Detection Completed!")


if __name__ == "__main__":
    main()