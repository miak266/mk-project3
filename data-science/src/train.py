
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""
Train and evaluate a used-car price prediction model.
"""

import argparse
from pathlib import Path

import pandas as pd
import mlflow
import mlflow.sklearn

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error


def parse_args():
    parser = argparse.ArgumentParser("train")

    parser.add_argument("--train_data", type=str, required=True)
    parser.add_argument("--test_data", type=str, required=True)
    parser.add_argument("--model_output", type=str, required=True)

    parser.add_argument(
        "--n_estimators",
        type=int,
        default=100
    )

    parser.add_argument(
        "--max_depth",
        type=int,
        default=10
    )

    return parser.parse_args()


def main(args):

    # Read training and testing datasets
    train_df = pd.read_csv(
        Path(args.train_data) / "train.csv"
    )

    test_df = pd.read_csv(
        Path(args.test_data) / "test.csv"
    )

    # Separate features and target
    y_train = train_df["price"]
    X_train = train_df.drop(columns=["price"])

    y_test = test_df["price"]
    X_test = test_df.drop(columns=["price"])

    # Create Random Forest regression model
    model = RandomForestRegressor(
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        random_state=42,
        n_jobs=-1
    )

    # Train the model
    model.fit(X_train, y_train)

    # Predict used-car prices
    predictions = model.predict(X_test)

    # Calculate Mean Squared Error
    mse = mean_squared_error(y_test, predictions)

    # Log model parameters
    mlflow.log_param("model", "RandomForestRegressor")
    mlflow.log_param("n_estimators", args.n_estimators)
    mlflow.log_param("max_depth", args.max_depth)

    # Log evaluation metric
    mlflow.log_metric("MSE", float(mse))

    print(f"Mean Squared Error: {mse:.4f}")

    # Save the trained model
    mlflow.sklearn.save_model(
        sk_model=model,
        path=args.model_output
    )

    print("Model training completed successfully.")


if __name__ == "__main__":

    args = parse_args()

    with mlflow.start_run():
        main(args)
