import warnings
warnings.filterwarnings("ignore")

import os
os.environ["DISABLE_PANDERA_IMPORT_WARNING"] = "True"

import pandas as pd
import pandera as pa
from pandera import Column, DataFrameSchema, Check


def define_schema():
    """TODO: Define Pandera schema"""

    return DataFrameSchema({

        "Type": Column(
            str,
            Check.isin(["L", "M", "H"]),
            nullable=False
        ),

        "Air temperature": Column(
            float,
            Check.between(295.0, 305.0),
            nullable=False
        ),

        "Process temperature": Column(
            float,
            Check.between(305.0, 315.0),
            nullable=False
        ),

        "Rotational speed": Column(
            "int64",
            Check.between(1000, 2900),
            nullable=False
        ),

        "Torque": Column(
            float,
            Check.between(3.0, 80.0),
            nullable=False
        ),

        "Tool wear": Column(
            "int64",
            Check.between(0, 253),
            nullable=False
        ),

        "Failure_Type": Column(
            "int64",
            Check.isin([0, 1, 2, 3, 4]),
            nullable=False
        ),
    })


def fix_dtypes(df):
    """Fix dataframe dtypes"""

    df = df.copy()

    df["Rotational speed"] = df["Rotational speed"].astype("int64")
    df["Tool wear"] = df["Tool wear"].astype("int64")
    df["Failure_Type"] = df["Failure_Type"].astype("int64")

    return df


def validate_train_current(train, current):
    """TODO: Validate train and current (must pass)"""

    schema = define_schema()

    train = fix_dtypes(train)
    current = fix_dtypes(current)

    schema.validate(train)
    schema.validate(current)

    print("Train and Current passed Pandera validation!")

    return train, current


def validate_stress(stress):
    """TODO: Validate stress with lazy=True (show all violations if any)"""

    schema = define_schema()

    stress = fix_dtypes(stress)

    try:
        schema.validate(stress, lazy=True)
        print("Stress passed Pandera validation!")

    except pa.errors.SchemaErrors as e:

        print("Stress validation failed. Summary of violations:")
        print(e.failure_cases)

    return stress


def main():

    print("\nRunning Data Validation Pipeline...\n")

    # TODO: Load datasets
    train = pd.read_csv("data/train.csv")
    current = pd.read_csv("data/current.csv")
    stress = pd.read_csv("data/stress.csv")

    # TODO: Validate train and current
    train, current = validate_train_current(train, current)

    # TODO: Validate stress
    stress = validate_stress(stress)

    print("\nData Validation Completed Successfully!")


if __name__ == "__main__":
    main()