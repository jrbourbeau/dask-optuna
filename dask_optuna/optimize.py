from typing import Callable

import optuna
from distributed import Client, default_client, wait

from .storage import DaskStorage

ObjectiveFuncType = Callable[[optuna.Trial], float]


def _optimize_batch(study_name, objective, storage=None, name=None, n_trials=None):
    dask_storage = DaskStorage(storage=storage, name=name)
    study = optuna.load_study(study_name=study_name, storage=dask_storage)
    study.optimize(objective, n_trials=n_trials)


def get_batch_sizes(n_trials, batch_size=None):
    if batch_size is None:
        # In not specified, use a single batch
        return [n_trials]

    batch_sizes = [batch_size] * (n_trials // batch_size)
    if n_trials % batch_size:
        batch_sizes += [n_trials % batch_size]
    return batch_sizes


def optimize(
    study: optuna.Study,
    func: ObjectiveFuncType,
    n_trials: int = 100,
    batch_size: int = None,
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
    batch_size
        Number of trials to perform per batch. Defaults to a single batch of size ``n_trials``.
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
            _optimize_batch,
            study_name=study.study_name,
            objective=func,
            storage=dask_storage.storage,
            name=dask_storage.name,
            n_trials=n,
            pure=False,
        )
        for n in get_batch_sizes(n_trials, batch_size)
    ]
    wait(futures)
