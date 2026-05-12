import pandas as pd

from evidently.legacy.report import Report
from evidently.legacy.metrics.data_drift.column_drift_metric import (
    ColumnDriftMetric
)


FEAT_COLS = [
    "Air temperature",
    "Process temperature",
    "Rotational speed",
    "Torque",
    "Tool wear"
]


def run_stress_drift_report(
    train,
    stress,
    save_path="drift_stress.html"
):
    """
    TODO: Run Evidently on stress batch with per-column metrics
    TODO: Save drift_stress.html
    TODO: Print per-column drift table
    """

    # TODO: Create metrics
    metrics = []

    for col in FEAT_COLS:
        metrics.append(
            ColumnDriftMetric(column_name=col)
        )

    # TODO: Create and run report
    drift_report_stress = Report(metrics=metrics)

    drift_report_stress.run(
        reference_data=train[FEAT_COLS],
        current_data=stress[FEAT_COLS]
    )

    # TODO: Save HTML report
    drift_report_stress.save_html(save_path)

    print(f"Saved drift report: {save_path}")

    # TODO: Convert report to dictionary
    report_dict = drift_report_stress.as_dict()

    # TODO: Create drift table
    rows = []

    for m in report_dict["metrics"]:

        res = m["result"]

        col = res["column_name"]

        drift_detected = res["drift_detected"]

        wasserstein_score = res["drift_score"]

        ref_mean = train[col].mean()

        cur_mean = stress[col].mean()

        delta = cur_mean - ref_mean

        rows.append({
            "feature": col,
            "drift_detected": drift_detected,
            "wasserstein_score": wasserstein_score,
            "ref_mean": ref_mean,
            "current_mean": cur_mean,
            "delta": delta
        })

    drift_table = pd.DataFrame(rows)

    drift_table = drift_table.sort_values(
        by="wasserstein_score",
        ascending=False
    )

    print("\nDrift Table (Stress Batch):")
    print(drift_table)

    return drift_report_stress, drift_table


def main():

    print("\nRunning Stress Batch Drift Detection...\n")

    # TODO: Load datasets
    train = pd.read_csv("data/train.csv")

    stress = pd.read_csv("data/stress.csv")

    # TODO: Run stress drift report
    drift_report_stress, drift_table = run_stress_drift_report(
        train=train,
        stress=stress,
        save_path="drift_stress.html"
    )

    print("\nStress Batch Drift Detection Completed!")


if __name__ == "__main__":
    main()