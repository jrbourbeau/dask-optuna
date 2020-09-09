from typing import Callable

import optuna
from distributed import Client, default_client, wait

from .storage import DaskStorage

ObjectiveFuncType = Callable[[optuna.Trial], float]


def run_trial(study_name, objective, storage_name=None):
    dask_storage = DaskStorage(name=storage_name)
    study = optuna.load_study(study_name=study_name, storage=dask_storage)
    study.optimize(objective, n_trials=1)


def optimize(
    study: optuna.Study,
    func: ObjectiveFuncType,
    n_trials: int = 100,
    client: Client = None,
) -> None:
    """Optimize an objective function

    Trials are batched and each batch is run in parllel on a Dask cluster.

    Parameters
    ----------
    study
        Optuna ``Study`` to use.
    func
        Objective function to optimize
    n_trials
        Number of optimization trials to perform. Defaults to 100.
    client
        Dask ``Client`` connected to your cluster. If not provided, ``dask.distributed.default_client()``
        if used to determine which ``Client`` should be used.
    """

    client = client or default_client()
    dask_storage = study._storage
    if not isinstance(dask_storage, DaskStorage):
        raise TypeError(
            f"Expected storage to be of type dask_optuna.DaskStorage but got {type(dask_storage)} instead"
        )

    futures = [
        client.submit(
            run_trial,
            study_name=study.study_name,
            objective=func,
            storage_name=dask_storage.name,
            pure=False,
        )
        for _ in range(n_trials)
    ]
    wait(futures)
