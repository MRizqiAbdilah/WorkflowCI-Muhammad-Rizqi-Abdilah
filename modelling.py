import argparse
import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import numpy as np
from mlflow.models.signature import infer_signature


def main(data_path):
    # =============================================================
    # MLflow setup (WAJIB untuk CI)
    # =============================================================
    mlflow.set_tracking_uri("file:./mlruns")
    mlflow.set_experiment("default")

    # Enable autolog BEFORE run
    mlflow.sklearn.autolog()

    with mlflow.start_run() as run:
        # Load data
        df = pd.read_csv(data_path)

        X = df.drop(columns=["TotalSpent_Bin", "Transaction Date", "Total Spent"])
        y = df["Total Spent"]

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        # Train model
        model = RandomForestRegressor(
            random_state=42,
            n_estimators=100
        )
        model.fit(X_train, y_train)

        # Predict
        preds = model.predict(X_test)

        # Metrics
        mse = mean_squared_error(y_test, preds)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_test, preds)
        r2 = r2_score(y_test, preds)

        # Log extra metrics (optional, tapi oke untuk tugas)
        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("mae", mae)
        mlflow.log_metric("r2_score", r2)

        # Signature
        signature = infer_signature(X_train, model.predict(X_train))

        # Log model (TANPA registry)
        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="model",
            signature=signature
        )

        print(f"Run ID: {run.info.run_id}")
        print(f"RMSE: {rmse:.4f}")
        print(f"MAE: {mae:.4f}")
        print(f"R2 Score: {r2:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default="clean_dataset.csv")
    args = parser.parse_args()

    main(args.data_path)
