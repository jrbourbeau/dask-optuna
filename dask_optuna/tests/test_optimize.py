import pytest
import optuna
from distributed import Client
from distributed.utils_test import gen_cluster

import dask_optuna
from dask_optuna.optimize import get_batch_sizes


def test_get_batch_sizes():
    assert get_batch_sizes(n_trials=10, batch_size=3) == [3, 3, 3, 1]
    assert get_batch_sizes(n_trials=10, batch_size=5) == [5, 5]
    assert get_batch_sizes(n_trials=10, batch_size=None) == [10]
    assert get_batch_sizes(n_trials=10, batch_size=10) == [10]


def objective(trial):
    x = trial.suggest_uniform("x", -10, 10)
    return (x - 2) ** 2


@pytest.mark.parametrize("processes", [True, False])
def test_optimize_sync(processes):
    with Client(processes=processes):
        study = optuna.create_study(storage=dask_optuna.DaskStorage())
        dask_optuna.optimize(
            study,
            objective,
            n_trials=10,
            batch_size=5,
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
            batch_size=5,
        )

    output = str(excinfo.value)
    assert "Expected" in output
    assert "dask_optuna.DaskStorage" in output
