import pytest
import optuna
from distributed import Client
from distributed.utils_test import gen_cluster

import dask_optuna
from .utils import get_storage_url


def objective(trial):
    x = trial.suggest_uniform("x", -10, 10)
    return (x - 2) ** 2


@pytest.mark.parametrize("storage_specifier", ["inmemory", "sqlite"])
@pytest.mark.parametrize("processes", [True, False])
def test_optimize_sync(storage_specifier, processes):
    with Client(processes=processes):
        with get_storage_url(storage_specifier) as url:
            storage = dask_optuna.DaskStorage(storage=url)
            study = optuna.create_study(storage=storage)
            dask_optuna.optimize(
                study,
                objective,
                n_trials=10,
            )
            assert len(study.trials) == 10


@gen_cluster(client=True)
def test_storage_raises(c, s, a, b):
    with pytest.raises(TypeError) as excinfo:
        study = optuna.create_study()
        dask_optuna.optimize(
            study,
            objective,
            n_trials=10,
        )

    output = str(excinfo.value)
    assert "Expected" in output
    assert "dask_optuna.DaskStorage" in output
