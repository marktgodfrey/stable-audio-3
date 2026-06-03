#!/usr/bin/env bash

set -euo pipefail

export AWS_REQUEST_CHECKSUM_CALCULATION="${AWS_REQUEST_CHECKSUM_CALCULATION:-when_required}"
export AWS_RESPONSE_CHECKSUM_VALIDATION="${AWS_RESPONSE_CHECKSUM_VALIDATION:-when_required}"

INTERVAL_SEC="${INTERVAL_SEC:-300}"
KEEP_LOCAL="${KEEP_LOCAL:-2}"
MIN_AGE_SEC="${MIN_AGE_SEC:-300}"
RUN_ONCE="${RUN_ONCE:-0}"
UPLOAD_RETRIES="${UPLOAD_RETRIES:-3}"

RUN_NAME="${RUN_NAME:-}"
RUN_ID="${RUN_ID:-}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-}"
S3_BASE="${S3_BASE:-}"

log() {
  echo "[$(date '+%Y-%m-%d %H:%M:%S %Z')] $*"
}

die() {
  log "ERROR: $*"
  exit 1
}

require_binary() {
  command -v "$1" >/dev/null 2>&1 || die "missing required binary: $1"
}

remote_size_bytes() {
  local remote_path="$1"
  aws s3 ls "$remote_path" 2>/dev/null | awk 'NF >= 4 {print $3}' | tail -n 1
}

remote_matches_local() {
  local local_path="$1"
  local remote_path="$2"
  local local_size remote_size

  [[ -f "${local_path}" ]] || return 1

  local_size="$(stat -c %s "$local_path" 2>/dev/null)" || return 1
  remote_size="$(remote_size_bytes "$remote_path")"

  [[ -n "${remote_size}" && "${remote_size}" == "${local_size}" ]]
}

write_latest_manifest() {
  local local_path="$1"
  local remote_path="$2"
  local s3_dir="$3"
  local fname step size uploaded_at tmp_manifest

  fname="$(basename "${local_path}")"
  step="$(sed -n 's/.*step=\([0-9][0-9]*\).*/\1/p' <<<"${fname}")"
  size="$(stat -c %s "${local_path}")"
  uploaded_at="$(date -u '+%Y-%m-%dT%H:%M:%SZ')"
  tmp_manifest="$(mktemp)"

  cat > "${tmp_manifest}" <<JSON
{
  "run_name": "${RUN_NAME}",
  "run_id": "${RUN_ID}",
  "checkpoint": "${fname}",
  "step": ${step:-null},
  "size_bytes": ${size},
  "s3_uri": "${remote_path}",
  "uploaded_at_utc": "${uploaded_at}"
}
JSON

  aws s3 cp "${tmp_manifest}" "${s3_dir}/latest.json" --only-show-errors
  rm -f "${tmp_manifest}"
}

latest_manifest_step() {
  local s3_dir="$1"
  local manifest

  manifest="$(aws s3 cp "${s3_dir}/latest.json" - 2>/dev/null || true)"
  [[ -n "${manifest}" ]] || {
    printf '0\n'
    return 0
  }

  python3 -c 'import json, sys; print(int(json.load(sys.stdin).get("step") or 0))' <<<"${manifest}" 2>/dev/null || printf '0\n'
}

cleanup_broken_symlinks() {
  local dir="$1"
  find "${dir}" -maxdepth 1 -type l -name 'last*.ckpt' ! -exec test -e {} \; -print | while IFS= read -r link_path; do
    log "removing broken symlink ${link_path}"
    rm -f "${link_path}"
  done
}

upload_checkpoint() {
  local local_path="$1"
  local remote_path="$2"
  local s3_dir="$3"
  local attempt

  if [[ ! -f "${local_path}" ]]; then
    log "skip upload; local checkpoint disappeared: ${local_path}"
    return 0
  fi

  if remote_matches_local "${local_path}" "${remote_path}"; then
    return 0
  fi

  log "uploading $(basename "${local_path}")"
  for ((attempt=1; attempt<=UPLOAD_RETRIES; attempt++)); do
    if aws s3 cp "${local_path}" "${remote_path}" --only-show-errors; then
      break
    fi

    if (( attempt == UPLOAD_RETRIES )); then
      die "upload failed after ${UPLOAD_RETRIES} attempt(s): ${local_path}"
    fi

    log "upload attempt ${attempt}/${UPLOAD_RETRIES} failed for $(basename "${local_path}"); retrying"
    sleep $((attempt * 10))
  done

  if remote_matches_local "${local_path}" "${remote_path}"; then
    log "uploaded $(basename "${local_path}")"
  else
    die "remote size mismatch after upload for ${local_path}"
  fi
}

write_latest_verified_manifest() {
  local dir="$1"
  local s3_dir="$2"
  local i fname local_path remote_path current_step candidate_step
  local -a checkpoints

  mapfile -t checkpoints < <(find "${dir}" -maxdepth 1 -type f -name 'epoch=*.ckpt' -printf '%f\n' | sort -V)
  current_step="$(latest_manifest_step "${s3_dir}")"

  for ((i=${#checkpoints[@]}-1; i>=0; i--)); do
    fname="${checkpoints[$i]}"
    local_path="${dir}/${fname}"
    remote_path="${s3_dir}/${fname}"

    if remote_matches_local "${local_path}" "${remote_path}"; then
      candidate_step="$(sed -n 's/.*step=\([0-9][0-9]*\).*/\1/p' <<<"${fname}")"
      if [[ -n "${candidate_step}" && "${candidate_step}" -lt "${current_step}" ]]; then
        log "latest manifest stays at step ${current_step}; newest verified local checkpoint is ${fname}"
        return 0
      fi

      write_latest_manifest "${local_path}" "${remote_path}" "${s3_dir}"
      log "latest manifest points to ${fname}"
      return 0
    fi
  done

  log "latest manifest not updated; no local checkpoint has a verified remote archive"
}

prune_old_checkpoints() {
  local dir="$1"
  local s3_dir="$2"
  local -a checkpoints
  local keep_cutoff i fname local_path remote_path

  mapfile -t checkpoints < <(find "${dir}" -maxdepth 1 -type f -name 'epoch=*.ckpt' -printf '%f\n' | sort -V)

  if (( ${#checkpoints[@]} <= KEEP_LOCAL )); then
    return 0
  fi

  keep_cutoff=$((${#checkpoints[@]} - KEEP_LOCAL))

  for ((i=0; i<keep_cutoff; i++)); do
    fname="${checkpoints[$i]}"
    local_path="${dir}/${fname}"
    remote_path="${s3_dir}/${fname}"

    if [[ ! -f "${local_path}" ]]; then
      log "skip prune; local checkpoint disappeared: ${fname}"
      continue
    fi

    if remote_matches_local "${local_path}" "${remote_path}"; then
      log "pruning local checkpoint ${fname}"
      rm -f "${local_path}"
    else
      log "skip prune for ${fname}; remote archive not confirmed"
    fi
  done

  cleanup_broken_symlinks "${dir}"
}

run_pass() {
  local dir="$1"
  local s3_dir="$2"
  local now mtime age local_path remote_path
  local -a checkpoints

  mapfile -t checkpoints < <(find "${dir}" -maxdepth 1 -type f -name 'epoch=*.ckpt' -printf '%f\n' | sort -V)

  if (( ${#checkpoints[@]} == 0 )); then
    log "no checkpoint files found in ${dir}"
    cleanup_broken_symlinks "${dir}"
    return 0
  fi

  log "found ${#checkpoints[@]} checkpoint file(s) in ${dir}"
  for fname in "${checkpoints[@]}"; do
    local_path="${dir}/${fname}"
    if [[ ! -f "${local_path}" ]]; then
      log "skip upload; local checkpoint disappeared: ${fname}"
      continue
    fi

    now="$(date +%s)"
    mtime="$(stat -c %Y "${local_path}" 2>/dev/null)" || {
      log "skip upload; could not stat checkpoint: ${fname}"
      continue
    }
    age=$((now - mtime))

    if (( age < MIN_AGE_SEC )); then
      log "skipping ${fname}; modified ${age}s ago"
      continue
    fi

    remote_path="${s3_dir}/${fname}"
    upload_checkpoint "${local_path}" "${remote_path}" "${s3_dir}"
  done

  prune_old_checkpoints "${dir}" "${s3_dir}"
  write_latest_verified_manifest "${dir}" "${s3_dir}"
}

main() {
  local dir s3_dir

  require_binary aws
  require_binary find
  require_binary stat

  [[ -n "${RUN_NAME}" ]] || die "RUN_NAME must be set"
  [[ -n "${RUN_ID}" ]] || die "RUN_ID must be set"
  [[ -n "${CHECKPOINT_DIR}" ]] || die "CHECKPOINT_DIR must be set"
  [[ -d "${CHECKPOINT_DIR}" ]] || die "CHECKPOINT_DIR does not exist: ${CHECKPOINT_DIR}"
  [[ -n "${S3_BASE}" ]] || die "S3_BASE must be set"

  dir="${CHECKPOINT_DIR}"
  s3_dir="${S3_BASE}/${RUN_NAME}/${RUN_ID}/checkpoints"

  log "checkpoint dir: ${dir}"
  log "s3 dir: ${s3_dir}"
  log "keep_local=${KEEP_LOCAL} min_age_sec=${MIN_AGE_SEC} interval_sec=${INTERVAL_SEC}"

  while true; do
    run_pass "${dir}" "${s3_dir}"

    if [[ "${RUN_ONCE}" == "1" ]]; then
      break
    fi

    sleep "${INTERVAL_SEC}"
  done
}

main "$@"
