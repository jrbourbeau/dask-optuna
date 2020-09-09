# Dask-Optuna

[![Tests](https://github.com/jrbourbeau/dask-optuna/workflows/Tests/badge.svg)](https://github.com/jrbourbeau/dask-optuna/actions?query=workflow%3ATests+branch%3Amaster)
[![Documentation](https://github.com/jrbourbeau/dask-optuna/workflows/Documentation/badge.svg)](https://github.com/jrbourbeau/dask-optuna/actions?query=workflow%3ADocumentation+branch%3Amaster)
[![Pre-commit](https://github.com/jrbourbeau/dask-optuna/workflows/Pre-commit/badge.svg)](https://github.com/jrbourbeau/dask-optuna/actions?query=workflow%3APre-commit+branch%3Amaster)

Dask-Optuna helps improve integration between [Optuna](https://optuna.org/) and [Dask](https://dask.org/)
by leveraging Optuna's existing distributed optimization capabilities to run
optimization trials in parallel on a Dask cluster. It does this by providing a
Dask-compatible `dask_optuna.DaskStorage` storage class which wraps an
Optuna storage class (e.g. Optuna's in-memory or sqlite storage) and can be used
directly by Optuna. For example:

```python
import optuna
import joblib
import dask.distributed
import dask_optuna

def objective(trial):
    x = trial.suggest_uniform("x", -10, 10)
    return (x - 2) ** 2

with dask.distributed.Client() as client:
    # Create a study using Dask-compatible storage
    storage = dask_optuna.DaskStorage()
    study = optuna.create_study(storage=storage)
    # Optimize in parallel on your Dask cluster
    with joblib.parallel_backend("dask"):
        study.optimize(objective, n_trials=100, n_jobs=-1)
    print(f"best_params = {study.best_params}")
```


## Documentation

See the [Dask-Optuna documentation](https://jrbourbeau.github.io/dask-optuna) for more information.


## License

[MIT License](LICENSE)
