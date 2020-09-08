import optuna
from distributed.utils_test import gen_cluster

import dask_optuna


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
