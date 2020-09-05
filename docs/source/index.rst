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

.. attention::

   This project is actively being developed so you may experience breaking
   changes without warning.


Example
-------

.. code-block::

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
      dask_optuna.optimize(study, objective, n_trials=100, batch_size=10)
      print(f"best_params = {study.best_params}")


Background
----------

Current Status
^^^^^^^^^^^^^^

Internally, Optuna uses `Joblib <https://joblib.readthedocs.io/en/latest/>`_ to support running optimizations in parallel.
Because Joblib has a Dask backend, you can seamlessly have Joblib execute tasks on a
Dask cluster by placing your Optuna code in a ``joblib.parallel_backend('dask')`` context manager:


.. code-block::

   import optuna
   from dask.distributed import Client
   import joblib

   client = Client(...)
   with joblib.parallel_backend('dask'):
      # Your optuna code here

This works very well for several use case.

Pain Points
^^^^^^^^^^^

However, there are pain points when it comes to using Optuna and Dask together today. In particular:

- Optuna's ``InMemory`` storage doesn’t support running optimization trials across multiple processes
  (https://github.com/optuna/optuna/issues/1232)
- Persistent storage (e.g. a sqlite database) is difficult to set up for a Dask cluster using remote workers,
  which may not have access to the same filesystem the cluster scheduler or client uses.
- ...

**Dask-Optuna helps address these pain points to improve scaling Optuna using Dask**

Community discussion
^^^^^^^^^^^^^^^^^^^^

Improving integration between Dask and Optuna has been discussed in both the
`Dask issue tracker <https://github.com/dask/dask/issues/6571>`_ and
`Optuna issue tracker <https://github.com/optuna/optuna/issues/1766>`_. Please feel free to join these discussions
if you'd like to get involved.
