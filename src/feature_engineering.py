import pandas as pd
import numpy as np


def engineer_features(df):
    """
    TODO: Compute engineered features
    """

    df = df.copy()

    # TODO: Compute Power_W
    df["Power_W"] = (
        df["Torque"]
        * (
            df["Rotational speed"]
            * 2
            * np.pi
            / 60
        )
    )

    # TODO: Compute Temp_diff
    df["Temp_diff"] = (
        df["Process temperature"]
        - df["Air temperature"]
    )

    return df


def print_grouped_feature_means(train):
    """
    TODO: Print grouped mean of engineered features
    """

    print(
        "\nMean of engineered features "
        "grouped by Failure_Type:\n"
    )

    grouped_means = train.groupby(
        "Failure_Type"
    )[["Power_W", "Temp_diff"]].mean()

    print(grouped_means)

    return grouped_means


def apply_feature_engineering(
    train,
    current,
    stress
):
    """
    Apply feature engineering to all datasets
    """

    print("\nApplying Feature Engineering...\n")

    # TODO: Engineer features
    train = engineer_features(train)

    current = engineer_features(current)

    stress = engineer_features(stress)

    # TODO: Print grouped means
    print_grouped_feature_means(train)

    print("\nFeature Engineering Completed!")

    return train, current, stress


def main():

    print("\nLoading datasets for Feature Engineering...\n")

    # TODO: Load datasets
    train = pd.read_csv("data/train.csv")

    current = pd.read_csv("data/current.csv")

    stress = pd.read_csv("data/stress.csv")

    # TODO: Apply feature engineering
    train, current, stress = apply_feature_engineering(
        train=train,
        current=current,
        stress=stress
    )


if __name__ == "__main__":
    main()