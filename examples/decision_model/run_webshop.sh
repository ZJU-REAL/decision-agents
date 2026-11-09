#!/usr/bin/env bash
set -euo pipefail

# Usage: run_webshop.sh [--dry-run] [vllm|sglang] [Hydra key=value ...]
# The default Teacher runs in process; select a real backend through Hydra overrides.
repo_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
cd "$repo_root"
PYTHON=${PYTHON:-python3}
dry_run=false
if [[ ${1:-} == --dry-run ]]; then
    dry_run=true
    shift
fi
ENGINE=vllm
if [[ $# -gt 0 && $1 != *=* && $1 != --* ]]; then
    ENGINE=$1
    shift
fi
case "$ENGINE" in vllm|sglang) ;; *) echo "Expected vllm or sglang" >&2; exit 2 ;; esac
context=${TEACHER_CONTEXT:-hindsight}
question=${TEACHER_QUESTION:-progress}
case "$context/$question" in hindsight/progress|hindsight/value|prefix/progress|prefix/value) ;;
    *) echo "Use TEACHER_CONTEXT=prefix|hindsight and TEACHER_QUESTION=progress|value" >&2; exit 2 ;;
esac
setting="${context}_${question}"
data_dir=${DATA_DIR:-$HOME/data/decision-models/webshop}
base_model=${BASE_MODEL:-Qwen/Qwen2.5-1.5B-Instruct}
extra_args=(-m verl.trainer.main_ppo)
if $dry_run; then
    extra_args+=(--cfg job --resolve)
fi
export VLLM_ATTENTION_BACKEND=XFORMERS

num_cpus_per_env_worker=0.1 # The CPU resource allocated for each environment worker. If you want to use less CPU resources, you can decrease this value.

train_data_size=16
val_data_size=128
group_size=8

# These rows only specify batch sizes; the environment samples actual tasks.
if ! $dry_run; then
    "$PYTHON" examples/decision_model/prepare_data.py \
        --output "$data_dir" --train-size "$train_data_size" --val-size "$val_data_size"
fi

exec "$PYTHON" "${extra_args[@]}" \
    algorithm.adv_estimator=grpo \
    decision_model.enable=True \
    decision_model.weight=0.5 \
    decision_model.fail_on_error=True \
    "decision_model.evaluations=[{name:${setting},model:fake_decisions_api,dataset:webshop,context:${context},question:${question},expansion:all_steps,metric:${question},options:{dataset:{max_steps:\${env.max_steps}},model:{}}}]" \
    "decision_model.advantage.options.query.config_name=${setting}" \
    "decision_model.advantage.options.query.metric=${question}" \
    "data.train_files='$data_dir/train.parquet'" \
    "data.val_files='$data_dir/test.parquet'" \
    data.train_batch_size=$train_data_size \
    data.val_batch_size=$val_data_size \
    data.max_prompt_length=4096 \
    data.max_response_length=512 \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    data.return_raw_chat=True \
    "actor_rollout_ref.model.path='$base_model'" \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=64 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=8 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.01 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=16 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=2 \
    "actor_rollout_ref.rollout.name=$ENGINE" \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
    actor_rollout_ref.rollout.enable_chunked_prefill=False \
    actor_rollout_ref.rollout.enforce_eager=False \
    actor_rollout_ref.rollout.free_cache_engine=False \
    actor_rollout_ref.rollout.val_kwargs.temperature=0.4 \
    actor_rollout_ref.rollout.val_kwargs.do_sample=True \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=16 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.use_invalid_action_penalty=True \
    actor_rollout_ref.actor.invalid_action_penalty_coef=0.1 \
    algorithm.use_kl_in_reward=False \
    env.env_name=Webshop \
    env.seed=0 \
    env.history_length=2 \
    env.max_steps=15 \
    env.rollout.n=$group_size \
    env.resources_per_worker.num_cpus=$num_cpus_per_env_worker \
    trainer.critic_warmup=0 \
    "trainer.logger=[console,wandb]" \
    trainer.project_name='decision_models_webshop' \
    "trainer.experiment_name=${setting}" \
    "trainer.n_gpus_per_node=${N_GPUS:-2}" \
    trainer.nnodes=1 \
    trainer.save_freq=50 \
    trainer.test_freq=5 \
    trainer.total_epochs=150 \
    trainer.total_training_steps=150 \
    trainer.resume_mode=disable \
    trainer.val_before_train=True "$@"
