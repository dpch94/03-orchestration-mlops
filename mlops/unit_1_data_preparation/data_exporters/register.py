import pandas as pd
import mlflow
from mlflow.entities import ViewType
from mlflow.tracking import MlflowClient
import mlflow.sklearn
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.metrics import root_mean_squared_error
import joblib
import os

if 'data_exporter' not in globals():
    from mage_ai.data_preparation.decorators import data_exporter

@data_exporter
def train(df: pd.DataFrame) -> pd.DataFrame:

    categorical = ['PULocationID', 'DOLocationID']
    df[categorical] = df[categorical].astype(str)
    train_dicts = df[categorical].to_dict(orient='records')
    dv = DictVectorizer()
    X_train = dv.fit_transform(train_dicts)

    target = 'duration'
    y_train = df[target].values
    lr = LinearRegression()
    lr.fit(X_train, y_train)

    # Make predictions
    y_pred = lr.predict(X_train)

    # Calculate RMSE
    rmse = mean_squared_error(y_train, y_pred, squared=False)
    print(f'Train RMSE: {rmse}')

    # Create artifacts directory if it doesn't exist
    artifacts_dir = "artifacts"
    os.makedirs(artifacts_dir, exist_ok=True)

    # Log the model and artifacts using MLflow
    with mlflow.start_run() as run:
        # Log the Linear Regression model
        mlflow.sklearn.log_model(lr, "linear_regression_model")

        # Save the DictVectorizer artifact
        dv_path = "dict_vectorizer.pkl"
        joblib.dump(dv, dv_path)

        # Log the DictVectorizer artifact
        mlflow.log_artifact(dv_path, artifact_path="artifacts")

        # Log parameters and metrics
        mlflow.log_param("model_type", "LinearRegression")
        mlflow.log_metric("rmse", rmse)

    return None
    
    