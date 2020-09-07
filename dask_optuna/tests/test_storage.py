import os
import tempfile

import pytest

import optuna
from distributed.utils_test import gen_cluster
from dask.distributed import wait

import dask_optuna


def objective(trial):
    x = trial.suggest_uniform("x", -10, 10)
    return (x - 2) ** 2


def _optimize(storage, name):
    dask_storage = dask_optuna.DaskStorage(storage=storage, name=name)
    study = optuna.create_study(
        study_name="foo", storage=dask_storage, load_if_exists=True
    )
    study.optimize(objective, n_trials=2)


@gen_cluster(client=True)
async def test_in_memory(c, s, a, b):
    storage = None
    dask_storage = await dask_optuna.DaskStorage(storage=storage)
    futures = [
        c.submit(
            _optimize, storage=dask_storage.storage, name=dask_storage.name, pure=False
        )
        for _ in range(5)
    ]
    await wait(futures)
    await futures[0]

    results = await dask_storage.get_all_study_summaries()
    assert len(results) == 1
    assert results[0].n_trials == 10


@pytest.mark.xfail(reason="TODO: fix me", strict=True)
@gen_cluster(client=True)
async def test_sqlite(c, s, a, b):
    with tempfile.TemporaryDirectory() as tmpdirname:
        storage = "sqlite:///" + os.path.join(tmpdirname, "example.db")

        dask_storage = await dask_optuna.DaskStorage(storage=storage)
        futures = [
            c.submit(
                _optimize,
                storage=dask_storage.storage,
                name=dask_storage.name,
                pure=False,
            )
            for _ in range(5)
        ]
        await wait(futures)
        await futures[0]

        results = await dask_storage.get_all_study_summaries()
        assert len(results) == 1
        assert results[0].n_trials == 10


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
