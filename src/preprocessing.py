from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE


FEATURES = [
    "Type_enc",
    "Air temperature",
    "Process temperature",
    "Rotational speed",
    "Torque",
    "Tool wear",
    "Power_W",
    "Temp_diff"
]


def encode_type_column(train, current, stress):
    """
    TODO:
    - Encode Type column using LabelEncoder
    - Add Type_enc column
    """

    le = LabelEncoder()

    train = train.copy()
    current = current.copy()
    stress = stress.copy()

    # =========================
    # TODO: Encode train data
    # =========================
    train["Type_enc"] = le.fit_transform(
        train["Type"]
    )

    # =========================
    # TODO: Encode current data
    # =========================
    current["Type_enc"] = le.transform(
        current["Type"]
    )

    # =========================
    # TODO: Encode stress data
    # =========================
    stress["Type_enc"] = le.transform(
        stress["Type"]
    )

    print("\nType column encoded successfully.")
    print("Encoding Mapping:")

    for idx, label in enumerate(le.classes_):
        print(f"{label} --> {idx}")

    return train, current, stress, le


def split_data(train):
    """
    TODO:
    - Define X and y
    - Perform train-validation split
    """

    # =========================
    # TODO: Define feature matrix
    # =========================
    X = train[FEATURES]

    # =========================
    # TODO: Define target column
    # =========================
    y = train["Failure_Type"]

    # =========================
    # TODO: Stratified train-validation split
    # =========================
    X_train, X_val, y_train, y_val = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    print("\nTrain-validation split completed.")

    print(f"X_train shape : {X_train.shape}")
    print(f"X_val shape   : {X_val.shape}")

    print(f"y_train shape : {y_train.shape}")
    print(f"y_val shape   : {y_val.shape}")

    return X_train, X_val, y_train, y_val


def apply_smote(X_train, y_train):
    """
    TODO:
    - Apply SMOTE on training data only
    """

    # =========================
    # TODO: Initialize SMOTE
    # =========================
    smote = SMOTE(
        k_neighbors=3,
        random_state=42
    )

    # =========================
    # TODO: Resample training data
    # =========================
    X_train_smote, y_train_smote = smote.fit_resample(
        X_train,
        y_train
    )

    print("\nSMOTE applied successfully.")

    print(f"Original training shape : {X_train.shape}")
    print(f"SMOTE training shape    : {X_train_smote.shape}")

    return X_train_smote, y_train_smote


def print_smote_distribution(y_train_smote):
    """
    TODO:
    - Print post-SMOTE class distribution
    """

    print("\n========== CLASS DISTRIBUTION AFTER SMOTE ==========")

    distribution = y_train_smote.value_counts().sort_index()

    print(distribution)


def prepare_training_data(train, current, stress):
    """
    Full preprocessing pipeline:
    - Encode categorical feature
    - Split dataset
    - Apply SMOTE
    """

    # =========================
    # TODO: Encode categorical column
    # =========================
    train, current, stress, le = encode_type_column(
        train,
        current,
        stress
    )

    # =========================
    # TODO: Split train-validation
    # =========================
    X_train, X_val, y_train, y_val = split_data(train)

    # =========================
    # TODO: Apply SMOTE
    # =========================
    X_train_smote, y_train_smote = apply_smote(
        X_train,
        y_train
    )

    # =========================
    # TODO: Print SMOTE distribution
    # =========================
    print_smote_distribution(y_train_smote)

    return (
        train,
        current,
        stress,
        X_train,
        X_val,
        y_train,
        y_val,
        X_train_smote,
        y_train_smote,
        le
    )


def main():
    """
    Main function for standalone execution
    """

    print("This module is intended to be imported into Jupyter Notebook.")
    print("Use prepare_training_data() after feature engineering.")


if __name__ == "__main__":
    main()