import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


CLASS_NAMES = {
    0: "No Failure",
    1: "TWF",
    2: "HDF",
    3: "PWF",
    4: "OSF"
}


def plot_failure_type_distribution(train, class_names):
    """
    TODO: Class distribution (print + bar chart) Failure_Type
    """

    class_counts = train["Failure_Type"].value_counts().sort_index()

    class_percent = (
        class_counts / len(train)
    ) * 100

    print("\nClass Distribution (Failure_Type):")

    for k, v in class_counts.items():

        print(
            f"{k} ({class_names[k]}): "
            f"{v} ({class_percent[k]:.2f}%)"
        )

    plt.figure(figsize=(8, 4))

    sns.barplot(
        x=class_counts.index,
        y=class_counts.values
    )

    plt.title("Failure_Type Class Distribution")

    plt.xlabel("Failure_Type")

    plt.ylabel("Count")

    plt.show()

    print("\nIs it balanced?")

    print(
        "-> If class counts are very different, "
        "the dataset is imbalanced."
    )


def plot_torque_distribution_by_failure(train):
    """
    TODO: Torque distribution by failure type
    (histogram, failures only)
    """

    failures_only = train[
        train["Failure_Type"] != 0
    ]

    plt.figure(figsize=(10, 5))

    sns.histplot(
        data=failures_only,
        x="Torque",
        hue="Failure_Type",
        bins=30,
        kde=True,
        element="step"
    )

    plt.title(
        "Torque Distribution by Failure Type "
        "(Failures Only)"
    )

    plt.xlabel("Torque")

    plt.ylabel("Count")

    plt.show()


def plot_tool_wear_distribution_by_failure(train):
    """
    TODO: Tool wear distribution by failure type
    (failures only)
    """

    failures_only = train[
        train["Failure_Type"] != 0
    ]

    plt.figure(figsize=(10, 5))

    sns.histplot(
        data=failures_only,
        x="Tool wear",
        hue="Failure_Type",
        bins=30,
        kde=True,
        element="step"
    )

    plt.title(
        "Tool wear Distribution by Failure Type "
        "(Failures Only)"
    )

    plt.xlabel("Tool wear")

    plt.ylabel("Count")

    plt.show()


def plot_type_distribution(train):
    """
    TODO: Type distribution (print + bar chart)
    """

    type_counts = train["Type"].value_counts()

    print("\nType Distribution:")

    print(type_counts)

    plt.figure(figsize=(6, 4))

    sns.barplot(
        x=type_counts.index,
        y=type_counts.values
    )

    plt.title("Type (L/M/H) Distribution")

    plt.xlabel("Type")

    plt.ylabel("Count")

    plt.show()


def run_complete_eda(train):
    """
    Run complete EDA pipeline
    """

    print("\nRunning Exploratory Data Analysis...\n")

    # TODO: Failure_Type distribution
    plot_failure_type_distribution(
        train=train,
        class_names=CLASS_NAMES
    )

    # TODO: Torque distribution
    plot_torque_distribution_by_failure(train)

    # TODO: Tool wear distribution
    plot_tool_wear_distribution_by_failure(train)

    # TODO: Type distribution
    plot_type_distribution(train)

    print("\nEDA Completed Successfully!")


def main():

    print("\nLoading dataset for EDA...\n")

    # TODO: Load dataset
    train = pd.read_csv("data/train.csv")

    # TODO: Run EDA
    run_complete_eda(train)


if __name__ == "__main__":
    main()