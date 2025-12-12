import argparse
import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

def main(data_path):
    # Set experiment (boleh tetap ada, aman)
    mlflow.set_experiment("default")

    # Tidak boleh start_run manual kalau pakai `mlflow run`
    mlflow.sklearn.autolog()

    df = pd.read_csv(data_path)
    X = df.drop(columns=["TotalSpent_Bin", "Transaction Date", "Total Spent"])
    y = df["Total Spent"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = RandomForestRegressor(random_state=42)
    model.fit(X_train, y_train)

    # log model (sekarang otomatis masuk ke run)
    mlflow.sklearn.log_model(model, name="model")

    preds = model.predict(X_test)
    mse = mean_squared_error(y_test, preds)

    # log metric
    mlflow.log_metric("mse", mse)

    print("MSE:", mse)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default="clean_dataset.csv")
    args = parser.parse_args()

    main(args.data_path)
