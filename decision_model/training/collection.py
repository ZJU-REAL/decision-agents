"""Capture explicit environment evidence before and after policy projection."""
from __future__ import annotations


def start_records(observation, trajectory_ids, max_steps):
    evidence = observation.get("decision_record")
    if evidence is None or len(evidence) != len(trajectory_ids):
        raise ValueError("enabled decision feedback requires manager decision_record evidence")
    return [dict(trajectory_id=str(trajectory_id), dataset=item["dataset"],
                 task=item["task"], max_steps=max_steps,
                 observations=[item["observation"]], actions=[], candidates=[],
                 policy_responses=[], extra={"rewards": [], "dones": [], "action_valid": []})
            for trajectory_id, item in zip(trajectory_ids, evidence)]


def pre_action_candidates(observation):
    # Detach mutable environment lists before env.step can change them in place.
    return tuple(tuple(item.get("candidates", ())) for item in observation["decision_record"])


def append_transition(records, next_observation, infos, responses, candidates,
                      rewards, dones, active):
    evidence = next_observation.get("decision_record")
    if evidence is None or len(evidence) != len(records):
        raise ValueError("missing post-action decision_record evidence")
    for i, record in enumerate(records):
        if not active[i]:
            continue
        action = infos[i].get("executed_action")
        if not isinstance(action, str):
            raise ValueError("manager must expose the actual post-projection executed_action")
        record["actions"].append(action)
        record["observations"].append(evidence[i]["observation"])
        record["candidates"].append(candidates[i])
        record["policy_responses"].append(responses[i])
        record["extra"]["rewards"].append(float(rewards[i]))
        record["extra"]["dones"].append(bool(dones[i]))
        record["extra"]["action_valid"].append(bool(infos[i].get("is_action_valid", True)))
