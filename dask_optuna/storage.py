from typing import Any, Dict, List, Optional
import uuid

import optuna
from optuna.distributions import (
    BaseDistribution,
    distribution_to_json,
    json_to_distribution,
)
from optuna import study
from optuna.trial import FrozenTrial
from optuna.trial import TrialState

from distributed import Client, get_worker
from distributed.utils import thread_state

from .serialize import (
    serialize_frozentrial,
    deserialize_frozentrial,
    serialize_studysummary,
    deserialize_studysummary,
)


class OptunaSchedulerExtension:
    def __init__(self, scheduler):
        self.scheduler = scheduler
        self.storages = {}

        self.scheduler.handlers.update(
            {
                "optuna_create_new_study": self.create_new_study,
                "optuna_delete_study": self.delete_study,
                "optuna_set_study_user_attr": self.set_study_user_attr,
                "optuna_set_study_system_attr": self.set_study_system_attr,
                "optuna_set_study_direction": self.set_study_direction,
                "optuna_get_study_id_from_name": self.get_study_id_from_name,
                "optuna_get_study_id_from_trial_id": self.get_study_id_from_trial_id,
                "optuna_get_study_name_from_id": self.get_study_name_from_id,
                "optuna_read_trials_from_remote_storage": self.read_trials_from_remote_storage,
                "optuna_get_study_direction": self.get_study_direction,
                "optuna_get_study_user_attrs": self.get_study_user_attrs,
                "optuna_get_study_system_attrs": self.get_study_system_attrs,
                "optuna_get_all_study_summaries": self.get_all_study_summaries,
                "optuna_create_new_trial": self.create_new_trial,
                "optuna_set_trial_state": self.set_trial_state,
                "optuna_set_trial_param": self.set_trial_param,
                "optuna_get_trial_number_from_id": self.get_trial_number_from_id,
                "optuna_get_trial_param": self.get_trial_param,
                "optuna_set_trial_value": self.set_trial_value,
                "optuna_set_trial_intermediate_value": self.set_trial_intermediate_value,
                "optuna_set_trial_user_attr": self.set_trial_user_attr,
                "optuna_set_trial_system_attr": self.set_trial_system_attr,
                "optuna_get_trial": self.get_trial,
                "optuna_get_all_trials": self.get_all_trials,
                "optuna_get_n_trials": self.get_n_trials,
            }
        )

        self.scheduler.extensions["optuna"] = self

    def get_storage(self, name):
        return self.storages[name]

    def create_new_study(
        self, comm, study_name: Optional[str] = None, storage_name: str = None
    ) -> int:
        return self.get_storage(storage_name).create_new_study(study_name=study_name)

    def delete_study(
        self, comm, study_id: int = None, storage_name: str = None
    ) -> None:
        return self.get_storage(storage_name).delete_study(study_id=study_id)

    def set_study_user_attr(
        self, comm, study_id: int, key: str, value: Any, storage_name: str = None
    ) -> None:
        return self.get_storage(storage_name).set_study_user_attr(
            study_id=study_id, key=key, value=value
        )

    def set_study_system_attr(
        self, comm, study_id: int, key: str, value: Any, storage_name: str = None
    ) -> None:
        return self.get_storage(storage_name).set_study_system_attr(
            study_id=study_id,
            key=key,
            value=value,
        )

    def set_study_direction(
        self,
        comm,
        study_id: int,
        direction: study.StudyDirection,
        storage_name: str = None,
    ) -> None:
        return self.get_storage(storage_name).set_study_direction(
            study_id=study_id,
            direction=direction,
        )

    def get_study_id_from_name(
        self, comm, study_name: str, storage_name: str = None
    ) -> int:
        return self.get_storage(storage_name).get_study_id_from_name(
            study_name=study_name
        )

    def get_study_id_from_trial_id(
        self, comm, trial_id: int, storage_name: str = None
    ) -> int:
        return self.get_storage(storage_name).get_study_id_from_trial_id(
            trial_id=trial_id
        )

    def get_study_name_from_id(
        self, comm, study_id: int, storage_name: str = None
    ) -> str:
        return self.get_storage(storage_name).get_study_name_from_id(study_id=study_id)

    def get_study_direction(
        self, comm, study_id: int, storage_name: str = None
    ) -> study.StudyDirection:
        return self.get_storage(storage_name).get_study_direction(study_id=study_id)

    def get_study_user_attrs(
        self, comm, study_id: int, storage_name: str = None
    ) -> Dict[str, Any]:
        return self.get_storage(storage_name).get_study_user_attrs(study_id=study_id)

    def get_study_system_attrs(
        self, comm, study_id: int, storage_name: str = None
    ) -> Dict[str, Any]:
        return self.get_storage(storage_name).get_study_system_attrs(study_id=study_id)

    def get_all_study_summaries(
        self, comm, storage_name: str = None
    ) -> List[study.StudySummary]:
        summaries = self.get_storage(storage_name).get_all_study_summaries()
        return [serialize_studysummary(s) for s in summaries]

    def create_new_trial(
        self,
        comm,
        study_id: int,
        template_trial: Optional[FrozenTrial] = None,
        storage_name: str = None,
    ) -> int:
        return self.get_storage(storage_name).create_new_trial(
            study_id=study_id,
            template_trial=template_trial,
        )

    def set_trial_state(
        self, comm, trial_id: int, state: TrialState, storage_name: str = None
    ) -> bool:
        return self.get_storage(storage_name).set_trial_state(
            trial_id=trial_id,
            state=getattr(TrialState, state),
        )

    def set_trial_param(
        self,
        comm,
        trial_id: int,
        param_name: str,
        param_value_internal: float,
        distribution: BaseDistribution,
        storage_name: str = None,
    ) -> None:
        distribution = json_to_distribution(distribution)
        return self.get_storage(storage_name).set_trial_param(
            trial_id=trial_id,
            param_name=param_name,
            param_value_internal=param_value_internal,
            distribution=distribution,
        )

    def get_trial_number_from_id(
        self, comm, trial_id: int, storage_name: str = None
    ) -> int:
        return self.get_storage(storage_name).get_trial_number_from_id(
            trial_id=trial_id
        )

    def get_trial_param(
        self, comm, trial_id: int, param_name: str, storage_name: str = None
    ) -> float:
        return self.get_storage(storage_name).get_trial_param(
            trial_id=trial_id,
            param_name=param_name,
        )

    def set_trial_value(
        self, comm, trial_id: int, value: float, storage_name: str = None
    ) -> None:
        return self.get_storage(storage_name).set_trial_value(
            trial_id=trial_id,
            value=value,
        )

    def set_trial_intermediate_value(
        self,
        comm,
        trial_id: int,
        step: int,
        intermediate_value: float,
        storage_name: str = None,
    ) -> None:
        return self.get_storage(storage_name).set_trial_intermediate_value(
            trial_id=trial_id,
            step=step,
            intermediate_value=intermediate_value,
        )

    def set_trial_user_attr(
        self, comm, trial_id: int, key: str, value: Any, storage_name: str = None
    ) -> None:
        return self.get_storage(storage_name).set_trial_user_attr(
            trial_id=trial_id,
            key=key,
            value=value,
        )

    def set_trial_system_attr(
        self, comm, trial_id: int, key: str, value: Any, storage_name: str = None
    ) -> None:
        return self.get_storage(storage_name).set_trial_system_attr(
            trial_id=trial_id,
            key=key,
            value=value,
        )

    def get_trial(self, comm, trial_id: int, storage_name: str = None) -> FrozenTrial:
        trial = self.get_storage(storage_name).get_trial(trial_id=trial_id)
        return serialize_frozentrial(trial)

    def get_all_trials(
        self, comm, study_id: int, deepcopy: bool = True, storage_name: str = None
    ) -> List[FrozenTrial]:
        trials = self.get_storage(storage_name).get_all_trials(
            study_id=study_id,
            deepcopy=deepcopy,
        )
        return [serialize_frozentrial(t) for t in trials]

    def get_n_trials(
        self,
        comm,
        study_id: int,
        state: Optional[TrialState] = None,
        storage_name: str = None,
    ) -> int:
        return self.get_storage(storage_name).get_n_trials(
            study_id=study_id,
            state=state,
        )

    def read_trials_from_remote_storage(
        self, comm, study_id: int, storage_name: str = None
    ) -> None:
        """Make an internal cache of trials up-to-date.
        Args:
            study_id:
                ID of the study.
        Raises:
            :exc:`KeyError`:
                If no study with the matching ``study_id`` exists.
        """
        return self.get_storage(storage_name).read_trials_from_remote_storage(
            study_id=study_id
        )


def register_with_scheduler(dask_scheduler=None, storage=None, name=None):
    if "optuna" not in dask_scheduler.extensions:
        ext = OptunaSchedulerExtension(dask_scheduler)
    else:
        ext = dask_scheduler.extensions["optuna"]

    if name not in ext.storages:
        ext.storages[name] = optuna.storages.get_storage(storage)


class DaskStorage(optuna.storages.BaseStorage):
    """ Implements Optuna Storage API """

    def __init__(self, storage=None, name=None, client=None):
        self.storage = storage
        self.name = name or f"dask-storage-{uuid.uuid4().hex}"
        try:
            self.client = client or Client.current()
        except ValueError:
            # Initialise new client
            self.client = get_worker().client

        if self.client.asynchronous or getattr(
            thread_state, "on_event_loop_thread", False
        ):

            async def _register():
                await self.client.run_on_scheduler(
                    register_with_scheduler, storage=self.storage, name=self.name
                )
                return self

            self.client.loop.add_callback(_register)
        else:
            self.client.run_on_scheduler(
                register_with_scheduler, storage=self.storage, name=self.name
            )

    def create_new_study(self, study_name: Optional[str] = None) -> int:
        return self.client.sync(
            self.client.scheduler.optuna_create_new_study,
            study_name=study_name,
            storage_name=self.name,
        )

    def delete_study(self, study_id: int) -> None:
        return self.client.sync(
            self.client.scheduler.optuna_delete_study,
            study_id=study_id,
            storage_name=self.name,
        )

    def set_study_user_attr(self, study_id: int, key: str, value: Any) -> None:
        return self.client.sync(
            self.client.scheduler.optuna_set_study_user_attr,
            study_id=study_id,
            key=key,
            value=value,
            storage_name=self.name,
        )

    def set_study_system_attr(self, study_id: int, key: str, value: Any) -> None:
        """Register an optuna-internal attribute to a study.

        This method overwrites any existing attribute.

        Args:
            study_id:
                ID of the study.
            key:
                Attribute key.
            value:
                Attribute value. It should be JSON serializable.

        Raises:
            :exc:`KeyError`:
                If no study with the matching ``study_id`` exists.
        """
        return self.client.sync(
            self.client.scheduler.optuna_set_study_system_attr,
            study_id=study_id,
            key=key,
            value=value,
            storage_name=self.name,
        )

    def set_study_direction(
        self, study_id: int, direction: study.StudyDirection
    ) -> None:
        """Register an optimization problem direction to a study.

        Args:
            study_id:
                ID of the study.
            direction:
                Either :obj:`~optuna.study.StudyDirection.MAXIMIZE` or
                :obj:`~optuna.study.StudyDirection.MINIMIZE`.

        Raises:
            :exc:`KeyError`:
                If no study with the matching ``study_id`` exists.
            :exc:`ValueError`:
                If the direction is already set and the passed ``direction`` is the opposite
                direction or :obj:`~optuna.study.StudyDirection.NOT_SET`.
        """
        return self.client.sync(
            self.client.scheduler.optuna_set_study_direction,
            study_id=study_id,
            direction=direction,
            storage_name=self.name,
        )

    # Basic study access

    def get_study_id_from_name(self, study_name: str) -> int:
        """Read the ID of a study.

        Args:
            study_name:
                Name of the study.

        Returns:
            ID of the study.

        Raises:
            :exc:`KeyError`:
                If no study with the matching ``study_name`` exists.
        """
        return self.client.sync(
            self.client.scheduler.optuna_get_study_id_from_name,
            study_name=study_name,
            storage_name=self.name,
        )

    def get_study_id_from_trial_id(self, trial_id: int) -> int:
        """Read the ID of a study to which a trial belongs.

        Args:
            trial_id:
                ID of the trial.

        Returns:
            ID of the study.

        Raises:
            :exc:`KeyError`:
                If no trial with the matching ``trial_id`` exists.
        """
        return self.client.sync(
            self.client.scheduler.optuna_get_study_id_from_trial_id,
            trial_id=trial_id,
            storage_name=self.name,
        )

    def get_study_name_from_id(self, study_id: int) -> str:
        """Read the study name of a study.

        Args:
            study_id:
                ID of the study.

        Returns:
            Name of the study.

        Raises:
            :exc:`KeyError`:
                If no study with the matching ``study_id`` exists.
        """
        return self.client.sync(
            self.client.scheduler.optuna_get_study_name_from_id,
            study_id=study_id,
            storage_name=self.name,
        )

    def get_study_direction(self, study_id: int) -> study.StudyDirection:
        """Read whether a study maximizes or minimizes an objective.

        Args:
            study_id:
                ID of a study.

        Returns:
            Optimization direction of the study.

        Raises:
            :exc:`KeyError`:
                If no study with the matching ``study_id`` exists.
        """
        return self.client.sync(
            self.client.scheduler.optuna_get_study_direction,
            study_id=study_id,
            storage_name=self.name,
        )

    def get_study_user_attrs(self, study_id: int) -> Dict[str, Any]:
        """Read the user-defined attributes of a study.

        Args:
            study_id:
                ID of the study.

        Returns:
            Dictionary with the user attributes of the study.

        Raises:
            :exc:`KeyError`:
                If no study with the matching ``study_id`` exists.
        """
        return self.client.sync(
            self.client.scheduler.optuna_get_study_user_attrs,
            study_id=study_id,
            storage_name=self.name,
        )

    def get_study_system_attrs(self, study_id: int) -> Dict[str, Any]:
        """Read the optuna-internal attributes of a study.

        Args:
            study_id:
                ID of the study.

        Returns:
            Dictionary with the optuna-internal attributes of the study.

        Raises:
            :exc:`KeyError`:
                If no study with the matching ``study_id`` exists.
        """
        return self.client.sync(
            self.client.scheduler.optuna_get_study_system_attrs,
            study_id=study_id,
            storage_name=self.name,
        )

    async def _get_all_study_summaries(self) -> List[study.StudySummary]:
        """Read a list of :class:`~optuna.study.StudySummary` objects.

        Returns:
            A list of :class:`~optuna.study.StudySummary` objects.

        """
        print(f"name = {self.name}")
        serialized_summaries = (
            await self.client.scheduler.optuna_get_all_study_summaries(
                storage_name=self.name
            )
        )
        return [deserialize_studysummary(s) for s in serialized_summaries]

    def get_all_study_summaries(self) -> List[study.StudySummary]:
        """Read a list of :class:`~optuna.study.StudySummary` objects.

        Returns:
            A list of :class:`~optuna.study.StudySummary` objects.

        """
        return self.client.sync(self._get_all_study_summaries)

    # Basic trial manipulation

    def create_new_trial(
        self, study_id: int, template_trial: Optional[FrozenTrial] = None
    ) -> int:
        """Create and add a new trial to a study.

        The returned trial ID is unique among all current and deleted trials.

        Args:
            study_id:
                ID of the study.
            template_trial:
                Template :class:`~optuna.trial.FronzenTrial` with default user-attributes,
                system-attributes, intermediate-values, and a state.

        Returns:
            ID of the created trial.

        Raises:
            :exc:`KeyError`:
                If no study with the matching ``study_id`` exists.
        """
        return self.client.sync(
            self.client.scheduler.optuna_create_new_trial,
            study_id=study_id,
            template_trial=template_trial,
            storage_name=self.name,
        )

    def set_trial_state(self, trial_id: int, state: TrialState) -> bool:
        """Update the state of a trial.

        Args:
            trial_id:
                ID of the trial.
            state:
                New state of the trial.

        Returns:
            :obj:`True` if the state is successfully updated.
            :obj:`False` if the state is kept the same.
            The latter happens when this method tries to update the state of
            :obj:`~optuna.trial.TrialState.RUNNING` trial to
            :obj:`~optuna.trial.TrialState.RUNNING`.

        Raises:
            :exc:`KeyError`:
                If no trial with the matching ``trial_id`` exists.
            :exc:`RuntimeError`:
                If the trial is already finished.
        """
        return self.client.sync(
            self.client.scheduler.optuna_set_trial_state,
            trial_id=trial_id,
            state=state.name,
            storage_name=self.name,
        )

    def set_trial_param(
        self,
        trial_id: int,
        param_name: str,
        param_value_internal: float,
        distribution: BaseDistribution,
    ) -> None:
        """Set a parameter to a trial.

        Args:
            trial_id:
                ID of the trial.
            param_name:
                Name of the parameter.
            param_value_internal:
                Internal representation of the parameter value.
            distribution:
                Sampled distribution of the parameter.

        Raises:
            :exc:`KeyError`:
                If no trial with the matching ``trial_id`` exists.
            :exc:`RuntimeError`:
                If the trial is already finished.
        """
        return self.client.sync(
            self.client.scheduler.optuna_set_trial_param,
            trial_id=trial_id,
            param_name=param_name,
            param_value_internal=param_value_internal,
            distribution=distribution_to_json(distribution),
            storage_name=self.name,
        )

    def get_trial_number_from_id(self, trial_id: int) -> int:
        """Read the trial number of a trial.

        .. note::

            The trial number is only unique within a study, and is sequential.

        Args:
            trial_id:
                ID of the trial.

        Returns:
            Number of the trial.

        Raises:
            :exc:`KeyError`:
                If no trial with the matching ``trial_id`` exists.
        """
        return self.client.sync(
            self.client.scheduler.optuna_get_trial_number_from_id,
            trial_id=trial_id,
            storage_name=self.name,
        )

    def get_trial_param(self, trial_id: int, param_name: str) -> float:
        """Read the parameter of a trial.

        Args:
            trial_id:
                ID of the trial.
            param_name:
                Name of the parameter.

        Returns:
            Internal representation of the parameter.

        Raises:
            :exc:`KeyError`:
                If no trial with the matching ``trial_id`` exists.
                If no such parameter exists.
        """
        return self.client.sync(
            self.client.scheduler.optuna_get_trial_param,
            trial_id=trial_id,
            param_name=param_name,
            storage_name=self.name,
        )

    def set_trial_value(self, trial_id: int, value: float) -> None:
        """Set a return value of an objective function.

        This method overwrites any existing trial value.

        Args:
            trial_id:
                ID of the trial.
            value:
                Value of the objective function.

        Raises:
            :exc:`KeyError`:
                If no trial with the matching ``trial_id`` exists.
            :exc:`RuntimeError`:
                If the trial is already finished.
        """
        return self.client.sync(
            self.client.scheduler.optuna_set_trial_value,
            trial_id=trial_id,
            value=value,
            storage_name=self.name,
        )

    def set_trial_intermediate_value(
        self, trial_id: int, step: int, intermediate_value: float
    ) -> None:
        """Report an intermediate value of an objective function.

        This method overwrites any existing intermediate value associated with the given step.

        Args:
            trial_id:
                ID of the trial.
            step:
                Step of the trial (e.g., the epoch when training a neural network).
            intermediate_value:
                Intermediate value corresponding to the step.

        Raises:
            :exc:`KeyError`:
                If no trial with the matching ``trial_id`` exists.
            :exc:`RuntimeError`:
                If the trial is already finished.
        """
        return self.client.sync(
            self.client.scheduler.optuna_set_trial_intermediate_value,
            trial_id=trial_id,
            step=step,
            intermediate_value=intermediate_value,
            storage_name=self.name,
        )

    def set_trial_user_attr(self, trial_id: int, key: str, value: Any) -> None:
        """Set a user-defined attribute to a trial.

        This method overwrites any existing attribute.

        Args:
            trial_id:
                ID of the trial.
            key:
                Attribute key.
            value:
                Attribute value. It should be JSON serializable.

        Raises:
            :exc:`KeyError`:
                If no trial with the matching ``trial_id`` exists.
            :exc:`RuntimeError`:
                If the trial is already finished.
        """
        return self.client.sync(
            self.client.scheduler.optuna_set_trial_user_attr,
            trial_id=trial_id,
            key=key,
            value=value,
            storage_name=self.name,
        )

    def set_trial_system_attr(self, trial_id: int, key: str, value: Any) -> None:
        """Set an optuna-internal attribute to a trial.

        This method overwrites any existing attribute.

        Args:
            trial_id:
                ID of the trial.
            key:
                Attribute key.
            value:
                Attribute value. It should be JSON serializable.

        Raises:
            :exc:`KeyError`:
                If no trial with the matching ``trial_id`` exists.
            :exc:`RuntimeError`:
                If the trial is already finished.
        """
        return self.client.sync(
            self.client.scheduler.optuna_set_trial_system_attr,
            trial_id=trial_id,
            key=key,
            value=value,
            storage_name=self.name,
        )

    # Basic trial access
    async def _get_trial(self, trial_id: int) -> FrozenTrial:
        serialized_trial = await self.client.scheduler.optuna_get_trial(
            trial_id=trial_id, storage_name=self.name
        )
        return deserialize_frozentrial(serialized_trial)

    def get_trial(self, trial_id: int) -> FrozenTrial:
        """Read a trial.

        Args:
            trial_id:
                ID of the trial.

        Returns:
            Trial with a matching trial ID.

        Raises:
            :exc:`KeyError`:
                If no trial with the matching ``trial_id`` exists.
        """

        return self.client.sync(self._get_trial, trial_id=trial_id)

    async def _get_all_trials(
        self, study_id: int, deepcopy: bool = True
    ) -> List[FrozenTrial]:
        serialized_trials = await self.client.scheduler.optuna_get_all_trials(
            study_id=study_id,
            deepcopy=deepcopy,
            storage_name=self.name,
        )
        return [deserialize_frozentrial(t) for t in serialized_trials]

    def get_all_trials(self, study_id: int, deepcopy: bool = True) -> List[FrozenTrial]:
        """Read all trials in a study.

        Args:
            study_id:
                ID of the study.
            deepcopy:
                Whether to copy the list of trials before returning.
                Set to :obj:`True` if you intend to update the list or elements of the list.

        Returns:
            List of trials in the study.

        Raises:
            :exc:`KeyError`:
                If no study with the matching ``study_id`` exists.
        """
        return self.client.sync(
            self._get_all_trials,
            study_id=study_id,
            deepcopy=deepcopy,
        )

    def get_n_trials(self, study_id: int, state: Optional[TrialState] = None) -> int:
        """Count the number of trials in a study.

        Args:
            study_id:
                ID of the study.
            state:
                :class:`~optuna.trial.TrialState` to filter trials.

        Returns:
            Number of trials in the study.

        Raises:
            :exc:`KeyError`:
                If no study with the matching ``study_id`` exists.
        """
        return self.client.sync(
            self.client.scheduler.optuna_get_n_trials,
            study_id=study_id,
            state=state,
            storage_name=self.name,
        )

    def read_trials_from_remote_storage(self, study_id: int) -> None:
        """Make an internal cache of trials up-to-date.
        Args:
            study_id:
                ID of the study.
        Raises:
            :exc:`KeyError`:
                If no study with the matching ``study_id`` exists.
        """
        return self.client.sync(
            self.client.scheduler.optuna_read_trials_from_remote_storage,
            study_id=study_id,
            storage_name=self.name,
        )
