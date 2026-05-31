#!/usr/bin/env bash
set -euo pipefail

repo_dir="${REPO_DIR:-/root/sa3/stable-audio-3}"
cd "$repo_dir"

run_name="${RUN_NAME:-sa3-instrumental750k}"
dataset_config="${DATASET_CONFIG:-$repo_dir/configs/dataset_configs/instrumental750k.json}"
model="${MODEL:-medium-base}"
model_config="${MODEL_CONFIG:-}"
checkpoint="${CHECKPOINT:-}"
resume_ckpt="${RESUME_CKPT:-$repo_dir/training_runs/$run_name/checkpoints/last.ckpt}"
save_dir="${SAVE_DIR:-$repo_dir/training_runs}"
export_path="${EXPORT_PATH:-}"
python_bin="${PYTHON_BIN:-python3}"

unset WANDB_RUN_ID
unset WANDB_RESUME
export WANDB_NAME="${WANDB_NAME:-$run_name}"
export PYTHONUNBUFFERED=1
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

log_file="${LOG_FILE:-$save_dir/$run_name/train.log}"
mkdir -p "$(dirname "$log_file")"

if [[ -n "${TMUX:-}" ]]; then
  tmux pipe-pane -o "cat >> '$log_file'"
  trap 'tmux pipe-pane >/dev/null 2>&1 || true' EXIT
else
  echo "This script should be run inside tmux for pipe-pane logging." >&2
  echo "Start one with: tmux new -s stable-audio-train" >&2
  exit 1
fi

echo "[train] start $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "[train] repo_dir=$repo_dir"
echo "[train] run_name=$run_name"
echo "[train] dataset_config=$dataset_config"

if [[ ! -f "$dataset_config" ]]; then
  echo "[train] missing dataset config: $dataset_config" >&2
  exit 1
fi

model_args=()
if [[ -n "$model_config" ]]; then
  model_args+=(--model_config "$model_config")
else
  model_args+=(--model "$model")
fi

if [[ -n "$checkpoint" ]]; then
  model_args+=(--checkpoint "$checkpoint")
fi

resume_args=()
if [[ -f "$resume_ckpt" ]]; then
  echo "[train] resuming trainer state from $resume_ckpt"
  resume_args+=(--resume_from_checkpoint "$resume_ckpt")
else
  echo "[train] no trainer resume checkpoint found at $resume_ckpt"
fi

export_args=()
if [[ -n "$export_path" ]]; then
  export_args+=(--export_path "$export_path")
fi

"$python_bin" scripts/train_diffusion.py \
  "${model_args[@]}" \
  --dataset_config "$dataset_config" \
  --name "$run_name" \
  --save_dir "$save_dir" \
  --seed "${SEED:-2025}" \
  --batch_size "${BATCH_SIZE:-1}" \
  --num_workers "${NUM_WORKERS:-8}" \
  --duration "${DURATION:-40}" \
  --steps "${STEPS:-1000000}" \
  --lr "${LR:-1e-5}" \
  --weight_decay "${WEIGHT_DECAY:-0.01}" \
  --accumulate_grad_batches "${ACCUMULATE_GRAD_BATCHES:-1}" \
  --gradient_clip_val "${GRADIENT_CLIP_VAL:-1.0}" \
  --checkpoint_every "${CHECKPOINT_EVERY:-500}" \
  --log_every "${LOG_EVERY:-100}" \
  --logger "${LOGGER:-csv}" \
  --precision "${PRECISION:-bf16-mixed}" \
  --strategy "${STRATEGY:-auto}" \
  "${resume_args[@]}" \
  "${export_args[@]}" || status=$?

status=${status:-0}
echo "[train] end $(date -u +%Y-%m-%dT%H:%M:%SZ) status=$status"
exit "$status"
