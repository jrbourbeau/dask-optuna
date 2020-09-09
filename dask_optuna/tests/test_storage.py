import pytest
import optuna
import joblib
from distributed import Client
from distributed.utils_test import gen_cluster

import dask_optuna
from .utils import get_storage_url


STORAGE_MODES = ["inmemory", "sqlite"]


def objective(trial):
    x = trial.suggest_uniform("x", -10, 10)
    return (x - 2) ** 2


@gen_cluster(client=True)
async def test_daskstorage_registers_extension(c, s, a, b):
    assert "optuna" not in s.extensions
    await dask_optuna.DaskStorage()
    assert "optuna" in s.extensions
    assert isinstance(s.extensions["optuna"], dask_optuna.OptunaSchedulerExtension)


@gen_cluster(client=True)
async def test_name(c, s, a, b):
    await dask_optuna.DaskStorage(name="foo")
    ext = s.extensions["optuna"]
    assert len(ext.storages) == 1
    assert isinstance(ext.storages["foo"], optuna.storages.InMemoryStorage)

    await dask_optuna.DaskStorage(name="bar")
    assert len(ext.storages) == 2
    assert isinstance(ext.storages["bar"], optuna.storages.InMemoryStorage)


@gen_cluster(client=True)
async def test_name_unique(c, s, a, b):
    s1 = await dask_optuna.DaskStorage()
    s2 = await dask_optuna.DaskStorage()
    assert s1.name != s2.name


@pytest.mark.parametrize("storage_specifier", STORAGE_MODES)
@pytest.mark.parametrize("processes", [True, False])
def test_optuna_joblib_backend(storage_specifier, processes):
    with Client(processes=processes):
        with get_storage_url(storage_specifier) as url:
            storage = dask_optuna.DaskStorage(url)
            study = optuna.create_study(storage=storage)
            with joblib.parallel_backend("dask"):
                study.optimize(objective, n_trials=10, n_jobs=-1)
            assert len(study.trials) == 10


@pytest.mark.parametrize("storage_specifier", STORAGE_MODES)
def test_get_base_storage(storage_specifier):
    with Client():
        with get_storage_url(storage_specifier) as url:
            dask_storage = dask_optuna.DaskStorage(url)
            storage = dask_storage.get_base_storage()
            expected_type = type(optuna.storages.get_storage(url))
            assert isinstance(storage, expected_type)
