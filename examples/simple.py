"""
Example to demonstrate using Dask-Optuna with Optuna's Joblib internals
to run optimization trials on a Dask cluster in parallel.
"""

import optuna
import joblib
from dask.distributed import Client
import dask_optuna

optuna.logging.set_verbosity(optuna.logging.WARN)


def objective(trial):
    x = trial.suggest_uniform("x", -10, 10)
    return (x - 2) ** 2


if __name__ == "__main__":

    with Client() as client:
        print(f"Dask dashboard is available at {client.dashboard_link}")
        dask_storage = dask_optuna.DaskStorage()
        study = optuna.create_study(storage=dask_storage)
        with joblib.parallel_backend("dask"):
            study.optimize(objective, n_trials=500, n_jobs=-1)

        print(f"best_params = {study.best_params}")
