import pandas as pd


CLASS_NAMES = {
    0: "No Failure",
    1: "TWF",
    2: "HDF",
    3: "PWF",
    4: "OSF"
}


def load_datasets(
    train_path="data/train.csv",
    current_path="data/current.csv",
    stress_path="data/stress.csv"
):
    """
    TODO: Load the three datasets
    """
    train = pd.read_csv(train_path)
    current = pd.read_csv(current_path)
    stress = pd.read_csv(stress_path)

    return train, current, stress


def print_dataset_shapes(train, current, stress):
    """
    TODO: Print dataset shapes
    """
    print(f"train  : {train.shape}")
    print(f"current: {current.shape}")
    print(f"stress : {stress.shape}")


def show_train_head(train, n=5):
    """
    TODO: Display first n rows of train dataset
    """
    return train.head(n)


def dataset_summary(train, current, stress):
    """
    TODO: Print dataset summary information
    """
    print("\n========== DATASET SUMMARY ==========")

    print("\nTrain Dataset:")
    print(train.info())

    print("\nCurrent Dataset:")
    print(current.info())

    print("\nStress Dataset:")
    print(stress.info())


def main():
    """
    Main function for dataset loading and inspection
    """

    # TODO: Load datasets
    train, current, stress = load_datasets()

    # TODO: Print dataset shapes
    print_dataset_shapes(train, current, stress)

    # TODO: Display first 5 rows of train dataset
    print("\nFirst 5 rows of Train Dataset:")
    print(show_train_head(train))

    # TODO: Print dataset summary
    dataset_summary(train, current, stress)


if __name__ == "__main__":
    main()