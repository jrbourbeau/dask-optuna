import datetime

from optuna.study import StudySummary, StudyDirection
from optuna.trial import FrozenTrial, TrialState
from optuna.distributions import distribution_to_json, json_to_distribution


def serialize_datetime(obj):
    if isinstance(obj, datetime.datetime):
        return {"__datetime__": True, "as_str": obj.strftime("%Y%m%dT%H:%M:%S.%f")}
    return obj


def deserialize_datetime(obj):
    if "__datetime__" in obj:
        obj = datetime.datetime.strptime(obj["as_str"], "%Y%m%dT%H:%M:%S.%f")
    return obj


def serialize_frozentrial(trial):
    data = trial.__dict__.copy()
    data["state"] = data["state"].name
    for attr in [
        "trial_id",
        "number",
        "params",
        "user_attrs",
        "system_attrs",
        "distributions",
        "datetime_start",
    ]:
        data[attr] = data.pop(f"_{attr}")
    data["distributions"] = {
        k: distribution_to_json(v) for k, v in data["distributions"].items()
    }
    data["datetime_start"] = serialize_datetime(data["datetime_start"])
    data["datetime_complete"] = serialize_datetime(data["datetime_complete"])
    return data


def deserialize_frozentrial(data):
    data["state"] = getattr(TrialState, data["state"])
    data["distributions"] = {
        k: json_to_distribution(v) for k, v in data["distributions"].items()
    }
    if data["datetime_start"] is not None:
        data["datetime_start"] = deserialize_datetime(data["datetime_start"])
    if data["datetime_complete"] is not None:
        data["datetime_complete"] = deserialize_datetime(data["datetime_complete"])
    trail = FrozenTrial(**data)
    return trail


def serialize_studysummary(summary):
    data = summary.__dict__.copy()
    data["study_id"] = data.pop("_study_id")
    data["best_trial"] = serialize_frozentrial(data["best_trial"])
    data["datetime_start"] = serialize_datetime(data["datetime_start"])
    data["direction"] = data["direction"]["name"]
    return data


def deserialize_studysummary(data):
    data["direction"] = getattr(StudyDirection, data["direction"])
    data["best_trial"] = deserialize_frozentrial(data["best_trial"])
    data["datetime_start"] = deserialize_datetime(data["datetime_start"])
    summary = StudySummary(**data)
    return summary
