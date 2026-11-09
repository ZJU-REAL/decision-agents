<h1 align="center">Actor, Judge, or Teacher?</h1>
<h3 align="center">Understanding Decision Models in Long-Horizon Agents</h3>

<p align="center">
  <a href="#abstract">Abstract</a> ·
  <a href="#overview">Overview</a> ·
  <a href="#main-takeaways">Main Takeaways</a> ·
  <a href="#installation">Installation</a> ·
  <a href="#training">Training</a>
</p>

<a id="abstract"></a>

## 📝 Abstract

Decision models are pretrained models that answer a question about a given context with a choice or a probability, so the same model can choose actions for an agent, judge actions already taken, or supervise a policy during training. However, a systematic understanding of how decision models behave across these roles, and how they should be queried in each, is still lacking. We fill this gap by studying three decision models and three general LLM baselines in all three roles, Actor, Judge, and Teacher, on ALFWorld and WebShop. As actors, all six models struggle when one goal condition must be met before another, often satisfying the easier condition first. As judges, they separate useful actions from actions that change nothing more clearly when asked whether an action helped achieve the goal than when asked whether the task will succeed. Showing the rest of the trajectory raises the scores of actions that temporarily undo progress, and decision models rank actions more accurately and at lower cost than their base LLMs prompted to answer the same questions. As a teacher, JEV improves policy training most when asked whether each action helped, with the full trajectory visible: a Qwen2.5-1.5B-Instruct policy improves over GRPO from 72.8% to 82.9% success on ALFWorld and from 56.8% to 71.4% on WebShop, and completes most tasks with ordering dependencies, on which JEV rarely succeeds as an actor. These results show that a decision model's ability to act does not fully reflect its value as a judge and teacher, which depends on the question it is asked and the part of the trajectory it sees.

<a id="overview"></a>

## 🔎 Overview

We study decision models in three roles, with different questions, available context, and uses of their answers:

- **Actor:** selects the next action from valid candidates using the goal and interaction history.
- **Judge:** scores recorded actions, varying the question (**Value** or **Progress**) and context (**Prefix** or **Hindsight**).
- **Teacher:** provides step-level judgments for policy training, combining them with outcome rewards.

<p align="center">
  <img src="assets/overview.png" alt="Decision models as Actors, Judges, and Teachers" width="100%">
</p>

<a id="main-takeaways"></a>

## 💡 Main Takeaways

- **Actors know the required step but are drawn to the easier one.** Actors choose the step that must come first when it is the only condition in the goal, and general reminders do not prevent the error when an easier condition is present. When a decision model must act, restructuring the goal, for example into ordered subgoals, is more promising than adding instructions to the prompt.
- **The question sets the granularity of the signal.** A question about the eventual outcome gives similar scores to all actions in a trajectory and largely repeats the outcome reward, while a question about each action's contribution separates actions within a trajectory. Before training, check that scores vary within trajectories.
- **Hindsight judgments target the actor's own failure.** With the rest of the trajectory visible, judges score premature actions lower and the recoveries that fix them higher. Decision models are therefore best used to score actions after an episode ends, as in RL training, and are less suited to filtering actions online, where only the prefix is available.
- **Judges can be screened before training.** The Judge measures on recorded trajectories identified the question and context that trained the best policy, at a small fraction of the cost of an RL run. Compare candidate judges and questions this way first, and choose by judging quality, not acting ability.
- **Step-level supervision becomes affordable.** Decision models judge more accurately and at lower cost than prompted general LLMs, and the Teacher is not needed at test time. Scoring every action during RL training is therefore practical, and can let a small policy outperform a larger one trained with outcome rewards alone.

<a id="repository-layout"></a>

## 🗂️ Repository Layout

```text
decision_model/
  actor.py                # Candidate-action selection
  judge.py                # Action scoring on recorded trajectories
  runner.py               # Actor/environment interaction
  models/                 # Decisions API, Prompted, and Constrained clients
  prompts/                # Dataset adapters, contexts, and questions
  advantage/              # Step-level advantages from model scores
  training/               # Teacher integration with policy training
examples/decision_model/
  run_alfworld.sh          # ALFWorld Teacher training
  run_webshop.sh           # WebShop Teacher training
  prepare_data.py         # Training data preparation
```

Everything else is inherited from the upstream `verl-agent` framework or kept as baseline/reproduction utilities.

<a id="installation"></a>

## 📦 Installation

Use separate environments for the base framework and WebShop, which requires Python ≤ 3.10.

### 1. Base Framework

```bash
conda create -n verl-agent python==3.12 -y
conda activate verl-agent

pip3 install torch==2.6.0 --index-url https://download.pytorch.org/whl/cu124
pip3 install flash-attn==2.7.4.post1 --no-build-isolation
pip3 install -e .
pip3 install vllm==0.8.5
```

### 2. ALFWorld

```bash
pip3 install gymnasium==0.29.1
pip3 install stable-baselines3==2.6.0
pip install alfworld
pip install vllm==0.8.5

alfworld-download -f
```

### 3. WebShop

```bash
conda create -n verl-agent-webshop python==3.10 -y
conda activate verl-agent-webshop

(cd agent_system/environments/env_package/webshop/webshop && bash setup.sh -d all)

pip3 install torch==2.6.0 --index-url https://download.pytorch.org/whl/cu124
pip3 install flash-attn==2.7.4.post1 --no-build-isolation
pip3 install -e .
pip3 install vllm==0.8.2
```

<a id="training"></a>

## 🚀 Training

Training scripts for **Qwen2.5-1.5B-Instruct** are provided in [`examples/decision_model/`](examples/decision_model/):

```bash
bash examples/decision_model/run_alfworld.sh    # ALFWorld
bash examples/decision_model/run_webshop.sh     # WebShop
```

<a id="acknowledgments"></a>

## 🙏 Acknowledgments

Our implementation uses [verl-agent / GiGPO](https://github.com/langfengQ/verl-agent) and its underlying [veRL](https://github.com/volcengine/verl) framework. We are grateful to their developers and to the teams behind [ALFWorld](https://github.com/alfworld/alfworld) and [WebShop](https://github.com/princeton-nlp/WebShop) for releasing their work.

Released under the [Apache License 2.0](LICENSE).
