# Dask-Optuna

[![Tests](https://github.com/jrbourbeau/dask-optuna/workflows/Tests/badge.svg)](https://github.com/jrbourbeau/dask-optuna/actions?query=workflow%3ATests+branch%3Amaster)
[![Pre-commit](https://github.com/jrbourbeau/dask-optuna/workflows/Pre-commit/badge.svg)](https://github.com/jrbourbeau/dask-optuna/actions?query=workflow%3APre-commit+branch%3Amaster)

Dask-Optuna helps improve integration between [Optuna](https://optuna.org/) and [Dask](https://dask.org/).

🚨🚨This project is actively being developed and is not yet ready for general use. Please do not use it. 🚨🚨

## Example

```python
import optuna
import dask.distributed
import dask_optuna

def objective(trial):
    x = trial.suggest_uniform("x", -10, 10)
    return (x - 2) ** 2

with dask.distributed.Client() as client:
    # Create a study using Dask-compatible storage
    study = optuna.create_study(storage=dask_optuna.DaskStorage())
    # Optimize in parallel on your Dask cluster
    dask_optuna.optimize(study, objective, n_trials=100)
    print(f"best_params = {study.best_params}")
```
