import argparse
import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import numpy as np

def main(data_path):
    # Start MLflow run
    with mlflow.start_run() as run:
        # Enable autolog for automatic logging
        mlflow.sklearn.autolog()

        # Load data
        df = pd.read_csv(data_path)
        X = df.drop(columns=["TotalSpent_Bin", "Transaction Date", "Total Spent"])
        y = df["Total Spent"]

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        # Train model
        model = RandomForestRegressor(random_state=42, n_estimators=100)
        model.fit(X_train, y_train)

        # Make predictions
        preds = model.predict(X_test)
        
        # Calculate metrics
        mse = mean_squared_error(y_test, preds)
        rmse = np.sqrt(mse)  # Calculate RMSE manually
        mae = mean_absolute_error(y_test, preds)
        r2 = r2_score(y_test, preds)

        # Log additional metrics (autolog already logs many metrics)
        mlflow.log_metric("mse", mse)
        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("mae", mae)
        mlflow.log_metric("r2_score", r2)

        # Log parameters
        mlflow.log_param("test_size", 0.2)
        mlflow.log_param("random_state", 42)

        # Explicitly log the model with signature for production use
        from mlflow.models.signature import infer_signature
        signature = infer_signature(X_train, model.predict(X_train))
        
        mlflow.sklearn.log_model(
            sk_model=model,
            name="model",
            signature=signature,
            registered_model_name="RandomForestRegressor"
        )

        print(f"Run ID: {run.info.run_id}")
        print(f"MSE: {mse:.4f}")
        print(f"RMSE: {rmse:.4f}")
        print(f"MAE: {mae:.4f}")
        print(f"R2 Score: {r2:.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default="clean_dataset.csv")
    args = parser.parse_args()

    main(args.data_path)