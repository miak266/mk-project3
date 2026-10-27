
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""
Prepare used-cars data for price prediction.
"""

import argparse
import os
import pandas as pd
import mlflow

from sklearn.model_selection import train_test_split


def parse_args():
    parser = argparse.ArgumentParser("prep")

    parser.add_argument("--raw_data", type=str, required=True)
    parser.add_argument("--train_data", type=str, required=True)
    parser.add_argument("--test_data", type=str, required=True)
    parser.add_argument("--test_train_ratio", type=float, default=0.2)

    return parser.parse_args()


def main(args):

    # Load used-cars dataset
    df = pd.read_csv(args.raw_data)

    print("Dataset columns:", df.columns.tolist())

    # Validate price column
    if "price" not in df.columns:
        raise ValueError("Column 'price' not found in dataset.")

    # Remove rows without a valid price
    df["price"] = pd.to_numeric(df["price"], errors="coerce")
    df = df.dropna(subset=["price"]).copy()

    # Separate features and target
    X = df.drop(columns=["price"])
    y = df["price"]

    # Split before preprocessing
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=args.test_train_ratio,
        random_state=42
    )

    # Encode categorical columns
    X_train = pd.get_dummies(X_train, dtype=int)
    X_test = pd.get_dummies(X_test, dtype=int)

    # Ensure identical columns
    X_test = X_test.reindex(
        columns=X_train.columns,
        fill_value=0
    )

    # Handle missing values using training medians
    X_train = X_train.apply(pd.to_numeric, errors="coerce")
    X_test = X_test.apply(pd.to_numeric, errors="coerce")

    medians = X_train.median().fillna(0)

    X_train = X_train.fillna(medians)
    X_test = X_test.fillna(medians)

    # Reattach target column
    train_df = X_train.copy()
    train_df["price"] = y_train.to_numpy()

    test_df = X_test.copy()
    test_df["price"] = y_test.to_numpy()

    # Save output files
    os.makedirs(args.train_data, exist_ok=True)
    os.makedirs(args.test_data, exist_ok=True)

    train_df.to_csv(
        os.path.join(args.train_data, "train.csv"),
        index=False
    )

    test_df.to_csv(
        os.path.join(args.test_data, "test.csv"),
        index=False
    )

    # Log dataset sizes
    mlflow.log_metric("train_size", len(train_df))
    mlflow.log_metric("test_size", len(test_df))

    print("Used-cars data preparation completed.")
    print("Training records:", len(train_df))
    print("Testing records:", len(test_df))


if __name__ == "__main__":
    args = parse_args()

    with mlflow.start_run():
        main(args)
