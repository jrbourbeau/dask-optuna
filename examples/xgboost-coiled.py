"""
Example using Coiled (https://coiled.io) and Dask-Optuna to run optimization trials
on a Dask cluster on AWS.

Here we use Optuna to tune hyperparameters for an XGBoost classifier.

Adapted from https://github.com/optuna/optuna/blob/master/examples/xgboost_simple.py
"""
from pprint import pprint

import numpy as np
import sklearn.datasets
import sklearn.metrics
from sklearn.model_selection import train_test_split
import xgboost as xgb

import optuna
import joblib
from dask.distributed import Client
import coiled
import dask_optuna

optuna.logging.set_verbosity(optuna.logging.WARN)


def objective(trial):
    X, y = sklearn.datasets.load_breast_cancer(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dtest = xgb.DMatrix(X_test, label=y_test)

    param = {
        "silent": 1,
        "objective": "binary:logistic",
        "booster": trial.suggest_categorical("booster", ["gbtree", "gblinear", "dart"]),
        "lambda": trial.suggest_float("lambda", 1e-8, 1.0, log=True),
        "alpha": trial.suggest_float("alpha", 1e-8, 1.0, log=True),
    }

    if param["booster"] == "gbtree" or param["booster"] == "dart":
        param["max_depth"] = trial.suggest_int("max_depth", 1, 9)
        param["eta"] = trial.suggest_float("eta", 1e-8, 1.0, log=True)
        param["gamma"] = trial.suggest_float("gamma", 1e-8, 1.0, log=True)
        param["grow_policy"] = trial.suggest_categorical(
            "grow_policy", ["depthwise", "lossguide"]
        )
    if param["booster"] == "dart":
        param["sample_type"] = trial.suggest_categorical(
            "sample_type", ["uniform", "weighted"]
        )
        param["normalize_type"] = trial.suggest_categorical(
            "normalize_type", ["tree", "forest"]
        )
        param["rate_drop"] = trial.suggest_float("rate_drop", 1e-8, 1.0, log=True)
        param["skip_drop"] = trial.suggest_float("skip_drop", 1e-8, 1.0, log=True)

    bst = xgb.train(param, dtrain)
    preds = bst.predict(dtest)
    pred_labels = np.rint(preds)
    accuracy = sklearn.metrics.accuracy_score(y_test, pred_labels)
    return accuracy


if __name__ == "__main__":
    with coiled.Cluster(n_workers=5, configuration="jrbourbeau/optuna") as cluster:
        with Client(cluster) as client:
            print(f"Dask dashboard is available at {client.dashboard_link}")

            storage = dask_optuna.DaskStorage("sqlite:///coiled-example.db")
            study = optuna.create_study(storage=storage, direction="maximize")
            with joblib.parallel_backend("dask"):
                study.optimize(objective, n_trials=100, n_jobs=-1)

            print("Best params:")
            pprint(study.best_params)
