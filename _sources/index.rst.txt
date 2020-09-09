Dask-Optuna
===========

.. toctree::
   :maxdepth: 1
   :hidden:

   install
   api
   changelog


Dask-Optuna helps improve integration between `Optuna <https://optuna.org/>`_
and `Dask <https://dask.org/>`_.


What Dask-Optuna does
---------------------

Dask-Optuna leverages Optuna's existing distributed optimization capabilities to run
optimization trials in parallel on a Dask cluster. It does this by providing a
Dask-compatible :class:`dask_optuna.DaskStorage` storage class which wraps an
Optuna storage class (e.g. Optuna's in-memory or sqlite storage) and can be used
directly by Optuna. For example:

.. code-block::

   import dask.distributed
   import dask_optuna
   
   client = dask.distributed.Client()
   # Wraps Optuna's in-memory storage
   storage_1 = dask_optuna.DaskStorage()
   # Wraps Optuna's SQLite DB storage
   storage_2 = dask_optuna.DaskStorage("sqlite:///example.db")

The underlying Optuna storage object lives on the cluster's scheduler and any method
calls on the ``DaskStorage`` instance results in the same method being called on the
underlying Optuna storage object.

This offers two primary benefits:

1. Helps extend Optuna's ``InMemoryStorage`` class to run across multiple processes.
   This is important when using remote workers in a Dask cluster or situations
   where Python's GIL leads to less-than-ideal parallelization.
2. Reduces setup when using persistent storage (e.g. creating a SQLite DB that's globally available)
   as the underlying Optuna storage class on the scheduler is accessible all workers
   in a Dask cluster.


Example
-------

.. code-block::

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


Community discussion
--------------------

Discussions on improving integration between Dask and Optuna are taking place in both the
`Dask issue tracker <https://github.com/dask/dask/issues/6571>`_ and
`Optuna issue tracker <https://github.com/optuna/optuna/issues/1766>`_. Please feel free to join
these conversations if you'd like to get involved.

If you have feedback or thoughts on how Dask-Optuna may be improved, please feel free to `open
an issue in Dask-Optuna's issue tracker <https://github.com/jrbourbeau/dask-optuna/issues/new>`_.


FAQ
---

When would I use this?
^^^^^^^^^^^^^^^^^^^^^^

Dask-Optuna is useful if you want to use Optuna's ``InMemoryStorage`` when running trials in
parallel across multiple processes or if the workers in your Dask cluster don't use the same
filesystem that your Dask ``Client`` uses. If, for example, you're using a
``dask.distributed.LocalCluster`` you may be better served by using Optuna's built in storage classes.