"""Two training entry points: score completed rollouts, then augment GRPO.

The batch owns all trajectory/score evidence. Dynamic collection explicitly
merges each scored shard before filtering. DataProto transformations carry only
row keys/score columns; its concat metadata behavior is never used to merge
independent evidence. apply() releases the store even on failure.
"""
from __future__ import annotations

import importlib
import inspect
import math
import uuid
from dataclasses import dataclass, field, replace
from typing import Any, Dict, List

from decision_model.advantage.base import (
    AdvantageConfig, RowData, RowRecordRef, Score, StepKey, StepRecord,
    compute_step_advantage,
)
from decision_model.advantage.view import InMemoryAdvantageData
from decision_model.config import EvaluationConfig
from decision_model.judge import Judge
from decision_model.registry import advantage_class, build_context_prompt, build_dataset_prompt, build_question_prompt
from decision_model.models.utils.loader import get_model
from decision_model.runtime import EvaluationRuntime
from decision_model.types import RequestSource

RECORDS_KEY = "decision_model_records"


@dataclass
class BatchRecords:
    batch_id: str
    trajectories: Dict[str, Any] = field(default_factory=dict)
    records: Dict[StepKey, StepRecord] = field(default_factory=dict)
    evaluations: List[Any] = field(default_factory=list)
    statistics: Dict[str, float] = field(default_factory=dict)
    closed: bool = False

    def merge(self, other):
        if self.closed or other.closed:
            raise ValueError("cannot merge a released decision batch")
        if self.batch_id != other.batch_id:
            raise ValueError("cannot merge evidence from another dynamic batch")
        if set(self.trajectories).intersection(other.trajectories):
            raise ValueError("duplicate trajectory identity in dynamic sampling")
        if set(self.records).intersection(other.records):
            raise ValueError("duplicate record identity in dynamic sampling")
        self.trajectories.update(other.trajectories)
        self.records.update(other.records)
        self.evaluations.extend(other.evaluations)
        for key, value in other.statistics.items():
            self.statistics[key] = self.statistics.get(key, 0) + value

    def clear(self):
        self.trajectories.clear()
        self.records.clear()
        self.evaluations.clear()
        self.statistics.clear()
        self.closed = True


def _get(value, name, default=None):
    return value.get(name, default) if hasattr(value, "get") else getattr(value, name, default)


def _import_components():
    # Models are loaded separately through the shared get_model() entry point.
    for name in ("prompts.datasets.alfworld",
                 "prompts.datasets.webshop", "prompts.contexts.hindsight", "prompts.contexts.prefix",
                 "prompts.questions.progress", "prompts.questions.value",
                 "advantage.direct", "advantage.group_centered"):
        importlib.import_module("decision_model." + name)


def build_training_bridge(config):
    """Build from root config; disabled/non-GRPO avoids model creation.

    Each evaluation selects a registered protocol or application backend by
    name. Its model options supply the user's served model, endpoint and any
    credential environment variable; the bridge supplies no provider defaults.
    """
    options = _get(config, "decision_model", {}) or {}
    estimator = _get(_get(config, "algorithm", {}), "adv_estimator", "")
    if not _get(options, "enable", False) or getattr(estimator, "value", estimator) != "grpo":
        return None
    _import_components()
    fail_on_error = _get(options, "fail_on_error", True)
    runtime_options = dict(_get(options, "runtime", {}) or {})
    judges = []
    for raw in _get(options, "evaluations", ()):
        config_eval = EvaluationConfig.from_dict(raw)
        component = config_eval.options
        model_options = dict(component.get("model") or {})
        model = get_model(config_eval.model, **model_options)
        runtime = EvaluationRuntime(model, cache_identity=model_options, **runtime_options)
        judges.append(Judge(
            config_eval, model=runtime,
            dataset_prompt=build_dataset_prompt(config_eval.dataset, **dict(component.get("dataset") or {})),
            context_prompt=build_context_prompt(config_eval.context, **dict(component.get("context") or {})),
            question_prompt=build_question_prompt(config_eval.question, **dict(component.get("question") or {})),
        ))
    advantage = AdvantageConfig.from_dict(_get(options, "advantage", {}))
    return TrainingBridge(judges, advantage, weight=_get(options, "weight", 1.0),
                          fail_on_error=fail_on_error)


class TrainingBridge:
    def __init__(self, judges, advantage, *, enabled=True, weight=1.0, fail_on_error=True):
        if not isinstance(weight, (int, float)) or isinstance(weight, bool) or not math.isfinite(weight):
            raise ValueError("decision step weight must be finite")
        if type(enabled) is not bool or type(fail_on_error) is not bool:
            raise ValueError("enabled and fail_on_error must be bool")
        self.judges = tuple(judges)
        if not self.judges:
            raise ValueError("enabled decision feedback requires at least one evaluation")
        names = [judge.config.name for judge in self.judges]
        if len(set(names)) != len(names):
            raise ValueError("evaluation names must be unique for row score columns")
        _import_components()
        # Failure policy is shared where an algorithm exposes that option. A
        # custom algorithm without it keeps its own constructor signature.
        parameters = inspect.signature(advantage_class(advantage.name).__init__).parameters
        if "fail_on_error" in parameters:
            algorithm_options = dict(advantage.options)
            if "fail_on_error" in algorithm_options and algorithm_options["fail_on_error"] != fail_on_error:
                raise ValueError("bridge and advantage fail_on_error policies conflict")
            algorithm_options["fail_on_error"] = fail_on_error
            advantage = AdvantageConfig(advantage.name, algorithm_options)
        self.advantage = advantage
        self.enabled = enabled
        self.weight = float(weight)
        self.fail_on_error = fail_on_error

    def enabled_for(self, adv_estimator, is_train=True):
        return self.enabled and is_train and getattr(adv_estimator, "value", adv_estimator) == "grpo"

    def new_batch(self):
        return BatchRecords("decision-training-" + uuid.uuid4().hex)

    def score_rollout(self, records, rows, *, batch_id):
        """Score at vanilla-rollout completion, before group filtering.

        ``rows`` is one list per trajectory and may include inactive trailing
        rows. All active rows must retain their collection-time step_idx.
        """
        if len(records) != len(rows):
            raise ValueError("trajectory records and row groups must align")
        shard = BatchRecords(batch_id)
        adapters = {e.config.dataset: e.dataset_prompt for e in self.judges}
        snapshots = []
        for raw in records:
            adapter = adapters.get(raw["dataset"])
            if adapter is None:
                raise ValueError("no judge dataset adapter for " + raw["dataset"])
            snapshot = adapter.adapt(raw)
            if snapshot.trajectory_id in shard.trajectories:
                raise ValueError("duplicate trajectory identity")
            shard.trajectories[snapshot.trajectory_id] = snapshot
            snapshots.append(snapshot)
            for k in range(snapshot.n_observations):
                key = StepKey(snapshot.trajectory_id, k)
                extra = {"trajectory": snapshot, "task": snapshot.task,
                         "terminal_boundary": k == snapshot.n_steps}
                for field_name in ("rewards", "dones", "action_valid"):
                    values = snapshot.extra.get(field_name)
                    if values is not None and k < len(values):
                        extra[field_name] = values[k]
                shard.records[key] = StepRecord(
                    key, snapshot.observations[k],
                    snapshot.actions[k] if k < snapshot.n_steps else None,
                    snapshot.observations[k + 1] if k < snapshot.n_steps else None,
                    snapshot.candidates[k] if k < len(snapshot.candidates) else (), extra=extra,
                )
        failures = []
        for judge in self.judges:
            selected = [s for s in snapshots if s.dataset == judge.config.dataset]
            before = judge.model.statistics() if hasattr(judge.model, "statistics") else {}
            try:
                evaluation = judge.evaluate(selected, batch_id=batch_id + "/" + uuid.uuid4().hex)
            finally:
                # Full request/response diagnostics belong to the explicit
                # archive/cache, not an implicit previous-training-batch buffer.
                judge.last_requests = []
                judge.last_results = []
            shard.evaluations.append(evaluation)
            after = judge.model.statistics() if hasattr(judge.model, "statistics") else {}
            for name, value in after.items():
                shard.statistics[name] = shard.statistics.get(name, 0) + value - before.get(name, 0)
            indexed = {}
            for item in evaluation.items:
                snapshot = shard.trajectories.get(item.trajectory_id)
                if snapshot is None:
                    raise ValueError("evaluation result references unknown trajectory")
                index = item.position.index if item.position.index is not None else snapshot.n_steps
                key = StepKey(item.trajectory_id, index)
                if key not in shard.records:
                    raise ValueError("evaluation position lies outside recorded trajectory")
                source = RequestSource(item.dataset, item.trajectory_id, item.position,
                                       item.config_name, item.batch_id)
                score = Score(item.metric, item.value, item.status, item.config_name, item.position_key, source)
                scores = dict(shard.records[key].scores)
                scores[item.request_id] = score
                shard.records[key] = replace(shard.records[key], scores=scores)
                if item.position.index is not None:
                    if key in indexed:
                        raise ValueError("one evaluation produced multiple row positions at the same step")
                    indexed[key] = item
                if not item.is_ok:
                    failures.append(item)
            for snapshot, trajectory_rows in zip(snapshots, rows):
                for row in trajectory_rows:
                    if str(row["traj_uid"]) != snapshot.trajectory_id:
                        raise ValueError("row trajectory does not match captured record")
                    step = row["step_idx"]
                    if isinstance(step, bool) or int(step) != step or int(step) < 0:
                        raise ValueError("step_idx must be a non-negative integer")
                    if row.get("active_masks", True) and int(step) >= snapshot.n_steps:
                        raise ValueError("active row has no executed record")
                    row["decision_batch_id"] = batch_id
                    item = indexed.get(StepKey(snapshot.trajectory_id, int(step)))
                    row["decision_score__" + judge.config.name] = item.value if item else None
                    row["decision_status__" + judge.config.name] = item.status if item else "unscored"
                    row["decision_position__" + judge.config.name] = item.position_key if item else None
        shard.statistics["evaluated_items"] = sum(len(e.items) for e in shard.evaluations)
        shard.statistics["failed_items"] = len(failures)
        if failures and self.fail_on_error:
            first = failures[0]
            raise RuntimeError("decision evaluation failed: {} {} {}".format(
                first.config_name, first.position_key, first.error))
        return shard

    def attach_batch(self, batch, store):
        if store.closed:
            raise ValueError("cannot attach a released batch")
        self._row_refs(batch, store)
        batch.meta_info[RECORDS_KEY] = store

    @staticmethod
    def _row_refs(batch, store):
        rows = batch.non_tensor_batch
        refs = []
        for i, (trajectory, step, batch_id) in enumerate(zip(
                rows["traj_uid"], rows["step_idx"], rows["decision_batch_id"])):
            if batch_id != store.batch_id:
                raise ValueError("row belongs to a different decision batch")
            if isinstance(step, bool) or int(step) != step or int(step) < 0:
                raise ValueError("step_idx must be a non-negative integer")
            key = StepKey(str(trajectory), int(step))
            if key not in store.records or store.records[key].action is None:
                raise KeyError("training row references missing action record {}".format(key))
            refs.append(RowRecordRef(i, key))
        if len(refs) != len(batch):
            raise ValueError("row mapping length differs from training batch")
        return refs

    def data_view(self, batch, *, multi_turn=False, history=None):
        store = batch.meta_info[RECORDS_KEY]
        if store.closed:
            raise ValueError("decision records were already released")
        refs = self._row_refs(batch, store)
        mask = grpo_mask(batch, multi_turn=multi_turn)
        valid = tuple(mask.bool().any(dim=-1).detach().cpu().tolist())
        values = dict(batch.non_tensor_batch)
        # Share existing tensor storage, including optional values/log-probs and
        # termination data; do not recompute absent training information.
        values.update({name: value for name, value in batch.batch.items()})
        reward = batch.non_tensor_batch.get("rewards")
        if reward is None:
            raise KeyError("environment rewards were not supplied")
        row_data = RowData(tuple(batch.non_tensor_batch["uid"]), tuple(reward), mask, valid, values)
        return InMemoryAdvantageData(store.records, refs, row_data, history=history,
                                     metadata={"batch_id": store.batch_id,
                                               "trajectories": store.trajectories,
                                               "evaluations": store.evaluations})

    def apply(self, batch, *, adv_estimator="grpo", is_train=True, multi_turn=False, history=None):
        if not self.enabled_for(adv_estimator, is_train):
            return {}
        store = batch.meta_info.get(RECORDS_KEY)
        if store is None:
            raise ValueError("enabled GRPO feedback has no collected decision records")
        try:
            data = self.data_view(batch, multi_turn=multi_turn, history=history)
            result = compute_step_advantage(data, self.advantage)
            if len(result.values) != len(batch):
                raise ValueError("step advantage must return exactly one value per training row")
            import numpy as np
            import torch
            advantages = batch.batch["advantages"]
            returns = batch.batch["returns"]
            mask = data.row_data.mask
            if advantages.shape != returns.shape or advantages.shape != mask.shape:
                raise ValueError("GRPO advantages, returns and actual token mask must have identical shapes")
            local = torch.as_tensor(result.values, dtype=advantages.dtype, device=advantages.device)
            if not torch.isfinite(local).all():
                raise ValueError("step advantage contains a non-finite value")
            if self.fail_on_error and any(status != "ok" for status in result.status):
                raise RuntimeError("step advantage failed")
            # Keep original scores separate from transformed row signal.
            batch.non_tensor_batch["decision_local"] = np.asarray(result.values, dtype=float)
            batch.non_tensor_batch["decision_local_status"] = np.asarray(result.status, dtype=object)
            if self.weight != 0:
                bonus = self.weight * local[:, None] * mask.to(dtype=advantages.dtype, device=advantages.device)
                # Both fields may share one GRPO tensor. Never mutate either input.
                batch.batch["advantages"] = advantages + bonus
                batch.batch["returns"] = returns + bonus.to(dtype=returns.dtype, device=returns.device)
            metrics = {"decision_model/" + name: value for name, value in store.statistics.items()}
            metrics.update({"decision_model/local_mean": float(local.mean().item()) if len(local) else 0.0,
                            "decision_model/rows": data.n_rows,
                            "decision_model/records": len(store.records)})
            return metrics
        finally:
            batch.meta_info.pop(RECORDS_KEY, None)
            store.clear()


def grpo_mask(batch, *, multi_turn=False):
    """Exactly the mask selection in ray_trainer.compute_advantage's GRPO arm."""
    mask = batch.batch["response_mask"]
    return batch.batch["loss_mask"][:, -mask.size(1):] if multi_turn else mask
