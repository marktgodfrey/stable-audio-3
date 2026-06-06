#!/usr/bin/env bash
set -euo pipefail

repo_dir="${REPO_DIR:-/root/sa3/stable-audio-3}"
cd "$repo_dir"

run_name="${RUN_NAME:-sa3-instrumental750k}"
dataset_config="${DATASET_CONFIG:-$repo_dir/configs/dataset_configs/instrumental750k.json}"
model="${MODEL:-medium-base}"
model_config="${MODEL_CONFIG:-}"
checkpoint="${CHECKPOINT:-}"
init_from_pretrained="${INIT_FROM_PRETRAINED:-false}"
pretransform_model="${PRETRANSFORM_MODEL:-}"
save_dir="${SAVE_DIR:-$repo_dir/training_runs}"
resume_ckpt="${RESUME_CKPT:-$save_dir/$run_name/checkpoints/last.ckpt}"
export_path="${EXPORT_PATH:-}"
python_bin="${PYTHON_BIN:-python3}"

if [[ -n "${WANDB_RUN_ID:-}" ]]; then
  export WANDB_RESUME="${WANDB_RESUME:-allow}"
else
  unset WANDB_RUN_ID
  unset WANDB_RESUME
fi
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
echo "[train] init_from_pretrained=$init_from_pretrained"

if [[ ! -f "$dataset_config" ]]; then
  echo "[train] missing dataset config: $dataset_config" >&2
  exit 1
fi

if "$python_bin" - "$dataset_config" <<'PY'
import json
import sys

with open(sys.argv[1]) as f:
    config = json.load(f)

if config.get("datasets_valid"):
    print(f"[train] validation datasets: {len(config['datasets_valid'])}")
else:
    print("[train] validation datasets: none")
PY
then
  :
else
  echo "[train] failed to inspect dataset config: $dataset_config" >&2
  exit 1
fi

model_args=()
if [[ -n "$model_config" ]]; then
  echo "[train] model_config=$model_config"
  model_args+=(--model_config "$model_config")
else
  echo "[train] model=$model"
  model_args+=(--model "$model")
fi

if [[ -n "$checkpoint" ]]; then
  model_args+=(--checkpoint "$checkpoint")
fi
if [[ "$init_from_pretrained" == "1" || "$init_from_pretrained" == "true" || "$init_from_pretrained" == "yes" ]]; then
  model_args+=(--init_from_pretrained)
else
  model_args+=(--no-init-from-pretrained)
fi
if [[ -n "$pretransform_model" ]]; then
  echo "[train] pretransform_model=$pretransform_model"
  model_args+=(--pretransform_model "$pretransform_model")
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

optimizer_args=()
if [[ "${IGNORE_MODEL_OPTIMIZER_CONFIG:-false}" == "1" || "${IGNORE_MODEL_OPTIMIZER_CONFIG:-false}" == "true" || "${IGNORE_MODEL_OPTIMIZER_CONFIG:-false}" == "yes" ]]; then
  echo "[train] optimizer_config=ignored"
  optimizer_args+=(--ignore_model_optimizer_config)
else
  echo "[train] optimizer_config=model_config_if_present"
fi

demo_args=()
if [[ -n "${DEMO_EVERY:-}" ]]; then
  demo_args+=(--demo_every "$DEMO_EVERY")
fi
if [[ -n "${DEMO_STEPS:-}" ]]; then
  demo_args+=(--demo_steps "$DEMO_STEPS")
fi
if [[ -n "${NUM_DEMOS:-}" ]]; then
  demo_args+=(--num_demos "$NUM_DEMOS")
fi
if [[ -n "${DEMO_CFG_SCALES:-}" ]]; then
  read -r -a demo_cfg_scales <<< "$DEMO_CFG_SCALES"
  demo_args+=(--demo_cfg_scales "${demo_cfg_scales[@]}")
fi
if [[ -n "${VALIDATION_EVERY:-}" ]]; then
  demo_args+=(--validation_every "$VALIDATION_EVERY")
fi
if [[ "${INPAINT_DEMOS_FROM_TRAIN_LOADER:-true}" == "0" || "${INPAINT_DEMOS_FROM_TRAIN_LOADER:-true}" == "false" || "${INPAINT_DEMOS_FROM_TRAIN_LOADER:-true}" == "no" ]]; then
  echo "[train] inpaint_demos_from_train_loader=false"
  demo_args+=(--no-inpaint-demos-from-train-loader)
elif [[ -n "${INPAINT_DEMOS_FROM_TRAIN_LOADER:-}" ]]; then
  echo "[train] inpaint_demos_from_train_loader=true"
  demo_args+=(--inpaint-demos-from-train-loader)
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
  --checkpoint_every "${CHECKPOINT_EVERY:-50000}" \
  --save_top_k "${SAVE_TOP_K:-10}" \
  --checkpoint_time_interval_minutes "${CHECKPOINT_TIME_INTERVAL_MINUTES:-60}" \
  --log_every "${LOG_EVERY:-100}" \
  --logger "${LOGGER:-wandb}" \
  --precision "${PRECISION:-bf16-mixed}" \
  --strategy "${STRATEGY:-auto}" \
  "${demo_args[@]}" \
  "${optimizer_args[@]}" \
  "${resume_args[@]}" \
  "${export_args[@]}" || status=$?

status=${status:-0}
echo "[train] end $(date -u +%Y-%m-%dT%H:%M:%SZ) status=$status"
exit "$status"
