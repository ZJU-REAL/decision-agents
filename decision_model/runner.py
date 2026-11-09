"""Single-episode execution against a caller-supplied environment adapter."""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Optional, Tuple

from decision_model.actor import Actor, ActorDecision
from decision_model.types import TaskSpec, TrajectorySnapshot


@dataclass(frozen=True)
class EnvironmentStep:
    """An environment observation and its currently executable candidates.

    Both reset() and step(action: str) return this type. The reset result may
    supply task text; run_episode's explicit task takes precedence. Adapters
    own environment-specific parsing and must execute the exact chosen command.
    """

    observation: str
    candidates: Tuple[str, ...]
    reward: float = 0.0
    done: bool = False
    task: Optional[str] = None

    def __post_init__(self):
        if not isinstance(self.observation, str):
            raise ValueError("observation must be a string")
        if isinstance(self.candidates, (str, bytes)):
            raise ValueError("candidates must be a sequence of commands")
        candidates = tuple(self.candidates)
        if any(not isinstance(command, str) or not command.strip() for command in candidates):
            raise ValueError("candidates must contain non-empty strings")
        object.__setattr__(self, "candidates", candidates)
        if (isinstance(self.reward, bool) or not isinstance(self.reward, (int, float))
                or not math.isfinite(self.reward)):
            raise ValueError("reward must be finite")
        if type(self.done) is not bool:
            raise ValueError("done must be bool")
        if self.task is not None and (not isinstance(self.task, str) or not self.task.strip()):
            raise ValueError("task must be non-empty text when supplied")


@dataclass(frozen=True)
class ActorEpisode:
    snapshot: Optional[TrajectorySnapshot]
    decisions: Tuple[ActorDecision, ...]
    status: str
    error: Optional[str] = None
    rewards: Tuple[float, ...] = ()

    @property
    def is_ok(self) -> bool:
        return self.status != "error"


def run_episode(env: Any, actor: Actor, *, dataset: str, trajectory_id: str,
                max_steps: int = 30, task: Optional[str] = None,
                task_id: Optional[str] = None, batch_id: str = "actor",
                timeout: Optional[float] = None) -> ActorEpisode:
    """Run an adapter implementing reset() and step(action) -> EnvironmentStep.

    The caller constructs and closes its environment and model. This loop does
    not import environments, infer dataset identities, or deploy services.
    Candidates are captured before each action; errors stop without fallback.
    """
    if type(max_steps) is not int or max_steps < 1:
        raise ValueError("max_steps must be a positive integer")
    for name, value in (("dataset", dataset), ("trajectory_id", trajectory_id)):
        if not isinstance(value, str) or not value.strip():
            raise ValueError(name + " must be a non-empty string")
    decisions = []
    rewards = []
    snapshot = None
    try:
        initial = env.reset()
        if not isinstance(initial, EnvironmentStep):
            raise TypeError("reset must return EnvironmentStep")
        task = task if task is not None else initial.task
        if not isinstance(task, str) or not task.strip():
            raise ValueError("supply task text explicitly or in the reset result")
        snapshot = TrajectorySnapshot(
            trajectory_id=trajectory_id, dataset=dataset,
            task=TaskSpec(task_id if task_id is not None else trajectory_id, task, max_steps),
            observations=(initial.observation,), actions=(), candidates=(initial.candidates,),
        )
    except Exception:
        return ActorEpisode(snapshot, (), "error", "environment_reset_error")
    if initial.done:
        return ActorEpisode(snapshot, (), "done")

    for _ in range(max_steps):
        decision = actor.act(snapshot, batch_id=batch_id, timeout=timeout)
        decisions.append(decision)
        if not decision.is_ok:
            return ActorEpisode(snapshot, tuple(decisions), "error", decision.error, tuple(rewards))
        try:
            step = env.step(decision.action)
            if not isinstance(step, EnvironmentStep):
                raise TypeError("step must return EnvironmentStep")
            snapshot = TrajectorySnapshot(
                trajectory_id=snapshot.trajectory_id, dataset=snapshot.dataset, task=snapshot.task,
                observations=snapshot.observations + (step.observation,),
                actions=snapshot.actions + (decision.action,),
                candidates=snapshot.candidates + (step.candidates,),
                policy_responses=snapshot.policy_responses, extra=snapshot.extra,
            )
            rewards.append(float(step.reward))
        except Exception:
            return ActorEpisode(snapshot, tuple(decisions), "error", "environment_step_error", tuple(rewards))
        if step.done:
            return ActorEpisode(snapshot, tuple(decisions), "done", rewards=tuple(rewards))
    return ActorEpisode(snapshot, tuple(decisions), "budget_exhausted", rewards=tuple(rewards))
