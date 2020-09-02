# Dask-Optuna

[![Tests](https://github.com/jrbourbeau/dask-optuna/workflows/Tests/badge.svg)](https://github.com/jrbourbeau/dask-optuna/actions?query=workflow%3ATests+branch%3Amaster)
[![Pre-commit](https://github.com/jrbourbeau/dask-optuna/workflows/Pre-commit/badge.svg)](https://github.com/jrbourbeau/dask-optuna/actions?query=workflow%3APre-commit+branch%3Amaster)

Dask-Optuna helps improve integration between [Optuna](https://optuna.org/) and [Dask](https://dask.org/).

🚨🚨This project is actively being developed and is not yet ready for general use. Please do not use it. 🚨🚨

## Example

```python
import optuna
from dask.distributed import Client, wait
import dask_optuna


def objective(trial):
    x = trial.suggest_uniform("x", -10, 10)
    return (x - 2) ** 2


def _optimize(storage=None):
    dask_storage = dask_optuna.DaskStorage(storage=storage)
    study = optuna.load_study(study_name="foo", storage=dask_storage)
    study.optimize(objective, n_trials=10)


with Client() as client:
    dask_storage = dask_optuna.DaskStorage()
    study = optuna.create_study(
        study_name="foo", storage=dask_storage, load_if_exists=True
    )
    futures = [client.submit(_optimize, pure=False) for _ in range(10)]
    wait(futures)

    study = optuna.load_study(study_name="foo", storage=dask_storage)
    print(f"study = {study}")
```
