import argparse
import gc
import hashlib
import json
import math
import multiprocessing as mp
import os
import queue
import re
import subprocess
import tarfile
import threading
import time
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from stable_audio_3 import AutoencoderModel


CAPTION_COLUMNS = ("dense_caption", "vivid_caption", "one_line_caption")
DROPPED_MANIFEST_COLUMNS = frozenset(("primary_genre_hint_id",))


def _import_pandas():
    try:
        import pandas as pd
    except ImportError as exc:
        raise ImportError(
            "pandas is required to read parquet manifests. "
            "Install pandas plus a parquet engine such as pyarrow or fastparquet."
        ) from exc
    return pd


def sanitize_path_part(value, fallback="unknown"):
    value = "" if value is None else str(value)
    value = value.strip()
    if not value:
        value = fallback
    value = re.sub(r"[^\w.\-]+", "_", value)
    return value[:200] or fallback


def to_jsonable(value):
    if isinstance(value, np.ndarray):
        return value.tolist()

    if hasattr(value, "item"):
        try:
            return to_jsonable(value.item())
        except Exception:
            pass

    if hasattr(value, "tolist") and not isinstance(value, (str, bytes, bytearray)):
        try:
            converted = value.tolist()
            if converted is not value:
                return to_jsonable(converted)
        except Exception:
            pass

    if isinstance(value, Path):
        return str(value)

    if isinstance(value, dict):
        return {str(k): to_jsonable(v) for k, v in value.items()}

    if isinstance(value, (list, tuple)):
        return [to_jsonable(v) for v in value]

    if isinstance(value, float) and math.isnan(value):
        return None

    if hasattr(value, "isoformat"):
        try:
            return value.isoformat()
        except Exception:
            pass

    return value


def build_output_base(row, sample_id_column, s3_uri_column):
    sample_id = row.get(sample_id_column)
    if sample_id is None:
        sample_id = Path(str(row.get(s3_uri_column, "sample"))).stem

    return Path(sanitize_path_part(sample_id, fallback="sample"))


def clean_manifest_row(row):
    return {
        k: to_jsonable(v)
        for k, v in row.items()
        if k not in DROPPED_MANIFEST_COLUMNS
    }


def get_non_empty_string(row, column):
    value = to_jsonable(row.get(column))
    if value is None:
        return None
    if isinstance(value, str):
        value = value.strip()
        return value or None
    return str(value)


def get_manifest_tempo(row):
    tempo = row.get("tempo")
    if tempo is None:
        return None
    try:
        return int(round(float(tempo)))
    except (TypeError, ValueError):
        return None


def build_base_conditioning_metadata(row):
    metadata = {}

    for column in CAPTION_COLUMNS:
        caption = get_non_empty_string(row, column)
        if caption is not None:
            metadata[column] = caption

    prompt = metadata.get("one_line_caption")
    if prompt is None:
        prompt = metadata.get("dense_caption") or metadata.get("vivid_caption")
    if prompt is not None:
        metadata["prompt"] = prompt

    tempo = get_manifest_tempo(row)
    if tempo is not None:
        metadata["tempo"] = tempo

    return metadata


def read_manifest(manifest_path):
    pd = _import_pandas()
    try:
        df = pd.read_parquet(manifest_path)
    except Exception as exc:
        raise RuntimeError(
            f"Failed to read parquet manifest at {manifest_path}. "
            "Make sure a parquet engine such as pyarrow or fastparquet is installed."
        ) from exc
    return df.to_dict("records")


def shard_rows(rows, num_shards, shard_index, start_row, limit):
    if shard_index < 0 or shard_index >= num_shards:
        raise ValueError("--shard-index must be in [0, --num-shards)")

    indexed_rows = list(enumerate(rows))
    indexed_rows = [item for item in indexed_rows if item[0] >= start_row]
    indexed_rows = [item for item in indexed_rows if item[0] % num_shards == shard_index]

    if limit is not None:
        indexed_rows = indexed_rows[:limit]

    return indexed_rows


def filter_existing_rows(indexed_rows, output_path, sample_id_column, s3_uri_column, completed_rows=None):
    done_bases = {
        p.name[:-9]
        for p in Path(output_path).glob("*.__done__")
    }
    if completed_rows:
        done_bases.update(str(x) for x in completed_rows)

    filtered_rows = []
    skipped = 0

    for row_index, row in indexed_rows:
        rel_base = str(build_output_base(row, sample_id_column, s3_uri_column))
        if rel_base in done_bases:
            skipped += 1
            continue
        filtered_rows.append((row_index, row))

    return filtered_rows, skipped


def assign_split(sample_id, val_ratio, split_seed, train_split_name, val_split_name):
    if val_ratio <= 0:
        return train_split_name
    if val_ratio >= 1:
        return val_split_name

    digest = hashlib.sha1(f"{split_seed}:{sample_id}".encode("utf-8")).digest()
    value = int.from_bytes(digest[:8], "big") / float(2 ** 64)
    return val_split_name if value < val_ratio else train_split_name


def sample_spool_dir(output_path, split_name):
    return Path(output_path) / "samples" / split_name


def sample_spool_base(output_path, split_name, rel_base, chunk_idx):
    return sample_spool_dir(output_path, split_name) / f"{rel_base}_{chunk_idx:03d}"


def shard_dir(output_path, split_name):
    return Path(output_path) / "shards" / split_name


def shard_state_path(output_path):
    return Path(output_path) / "shard_state.json"


def aws_cp_args(src, dst, aws_profile=None, request_payer=False):
    cmd = ["aws", "s3", "cp", src, dst]
    if aws_profile:
        cmd.extend(["--profile", aws_profile])
    if request_payer:
        cmd.extend(["--request-payer", "requester"])
    return cmd


def load_shard_state(output_path, split_names):
    state_path = shard_state_path(output_path)
    if state_path.exists():
        try:
            state = json.loads(state_path.read_text())
        except Exception:
            state = {}
    else:
        state = {}

    shard_indices = state.get("shard_indices", {})
    for split_name in split_names:
        shard_indices.setdefault(split_name, 0)
    state["shard_indices"] = shard_indices
    completed_rows = state.get("completed_rows", [])
    if not isinstance(completed_rows, list):
        completed_rows = []
    state["completed_rows"] = completed_rows
    return state


def save_shard_state(output_path, state):
    state_path = shard_state_path(output_path)
    state_path.write_text(json.dumps(state, indent=2))


def merge_shard_states(base_state, other_state, split_names):
    merged = {
        "shard_indices": {},
        "completed_rows": sorted(
            set(base_state.get("completed_rows", [])) | set(other_state.get("completed_rows", []))
        ),
    }
    for split_name in split_names:
        merged["shard_indices"][split_name] = max(
            int(base_state.get("shard_indices", {}).get(split_name, 0)),
            int(other_state.get("shard_indices", {}).get(split_name, 0)),
        )
    return merged


class ShardWriterManager:
    def __init__(
        self,
        output_path,
        split_names,
        samples_per_shard,
        s3_prefix=None,
        aws_profile=None,
        request_payer=False,
        keep_local_shards=False,
        keep_local_samples=False,
        state_s3_uri=None,
        shard_name_prefix="",
    ):
        self.output_path = Path(output_path)
        self.split_names = tuple(split_names)
        self.samples_per_shard = int(samples_per_shard)
        self.s3_prefix = s3_prefix.rstrip("/") if s3_prefix else None
        self.aws_profile = aws_profile
        self.request_payer = request_payer
        self.keep_local_shards = keep_local_shards
        self.keep_local_samples = keep_local_samples
        self.shard_name_prefix = shard_name_prefix
        self.pending_samples = {split_name: [] for split_name in self.split_names}
        self.state_s3_uri = state_s3_uri
        self.state = load_shard_state(self.output_path, self.split_names)
        self.state = self._merge_remote_state(self.state)

        for split_name in self.split_names:
            sample_spool_dir(self.output_path, split_name).mkdir(parents=True, exist_ok=True)
            shard_dir(self.output_path, split_name).mkdir(parents=True, exist_ok=True)

        save_shard_state(self.output_path, self.state)

    def _merge_remote_state(self, local_state):
        if self.state_s3_uri is None:
            return local_state

        remote_tmp = self.output_path / ".remote_shard_state.json"
        try:
            subprocess.run(
                aws_cp_args(
                    self.state_s3_uri,
                    str(remote_tmp),
                    aws_profile=self.aws_profile,
                    request_payer=self.request_payer,
                ),
                check=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            remote_state = json.loads(remote_tmp.read_text())
            merged = merge_shard_states(local_state, remote_state, self.split_names)
            print(
                f"[state] merged remote state from {self.state_s3_uri}: "
                f"completed_rows={len(merged.get('completed_rows', []))}"
            )
            return merged
        except Exception:
            return local_state
        finally:
            if remote_tmp.exists():
                remote_tmp.unlink()

    def _persist_state(self):
        save_shard_state(self.output_path, self.state)
        if self.state_s3_uri is None:
            return
        subprocess.run(
            aws_cp_args(
                str(shard_state_path(self.output_path)),
                self.state_s3_uri,
                aws_profile=self.aws_profile,
                request_payer=self.request_payer,
            ),
            check=True,
        )

    def recover_local_samples(self):
        recovered = 0
        for split_name in self.split_names:
            split_dir = sample_spool_dir(self.output_path, split_name)
            sample_map = {}
            for path in split_dir.glob("*"):
                if path.suffix not in {".npy", ".json"}:
                    continue
                sample_map.setdefault(path.stem, {})[path.suffix] = path

            recovered_records = []
            for stem, files in sorted(sample_map.items()):
                if ".npy" not in files or ".json" not in files:
                    continue
                recovered_records.append(
                    {
                        "row_base": row_base_from_sample_stem(stem),
                        "npy_path": files[".npy"],
                        "json_path": files[".json"],
                        "npy_arcname": f"{stem}.npy",
                        "json_arcname": f"{stem}.json",
                    }
                )
            if recovered_records:
                print(f"[recover:{split_name}] found {len(recovered_records)} local samples waiting to be sharded")
                self.pending_samples.setdefault(split_name, []).extend(recovered_records)
                recovered += len(recovered_records)
                while len(self.pending_samples[split_name]) >= self.samples_per_shard:
                    self.flush_split(split_name)
        return recovered

    def add_samples(self, split_name, sample_records):
        pending = self.pending_samples.setdefault(split_name, [])
        if pending and len(pending) + len(sample_records) > self.samples_per_shard:
            self.flush_split(split_name, max_samples=self.samples_per_shard)
            pending = self.pending_samples.setdefault(split_name, [])
        pending.extend(sample_records)
        while len(self.pending_samples[split_name]) >= self.samples_per_shard:
            self.flush_split(split_name, max_samples=self.samples_per_shard)

    def flush_split(self, split_name, max_samples=None):
        samples = self.pending_samples.get(split_name, [])
        if not samples:
            return None

        if max_samples is None or max_samples >= len(samples):
            to_flush = list(samples)
            self.pending_samples[split_name] = []
        else:
            to_flush = list(samples[:max_samples])
            self.pending_samples[split_name] = list(samples[max_samples:])

        shard_idx = int(self.state["shard_indices"].get(split_name, 0))
        out_path = shard_dir(self.output_path, split_name) / f"{self.shard_name_prefix}shard-{shard_idx:06d}.tar"

        with tarfile.open(out_path, "w") as tar:
            for sample in to_flush:
                tar.add(sample["npy_path"], arcname=sample["npy_arcname"], recursive=False)
                tar.add(sample["json_path"], arcname=sample["json_arcname"], recursive=False)

        self.state["shard_indices"][split_name] = shard_idx + 1
        self.upload_and_cleanup(split_name, out_path, to_flush)
        completed = set(self.state.get("completed_rows", []))
        completed.update(
            sample.get("row_base") or row_base_from_sample_stem(Path(sample["npy_arcname"]).stem)
            for sample in to_flush
        )
        self.state["completed_rows"] = sorted(completed)
        self._persist_state()
        return out_path

    def upload_and_cleanup(self, split_name, shard_path, samples):
        if self.s3_prefix is not None:
            destination = f"{self.s3_prefix}/{split_name}/{shard_path.name}"
            cmd = ["aws", "s3", "cp", str(shard_path), destination]
            if self.aws_profile:
                cmd.extend(["--profile", self.aws_profile])
            if self.request_payer:
                cmd.extend(["--request-payer", "requester"])
            print(f"[upload:{split_name}] {shard_path} -> {destination}")
            subprocess.run(cmd, check=True)

        if not self.keep_local_samples:
            for sample in samples:
                if sample["npy_path"].exists():
                    sample["npy_path"].unlink()
                if sample["json_path"].exists():
                    sample["json_path"].unlink()

        if self.s3_prefix is not None and not self.keep_local_shards and shard_path.exists():
            shard_path.unlink()
            print(f"[upload:{split_name}] removed local shard {shard_path}")

    def flush_all(self):
        for split_name in self.split_names:
            self.flush_split(split_name)


def start_audio_stream(s3_uri, sample_rate, channels, aws_profile=None, request_payer=False):
    aws_cmd = ["aws", "s3", "cp", s3_uri, "-"]
    if aws_profile:
        aws_cmd.extend(["--profile", aws_profile])
    if request_payer:
        aws_cmd.extend(["--request-payer", "requester"])

    ffmpeg_cmd = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        "pipe:0",
        "-f",
        "f32le",
        "-acodec",
        "pcm_f32le",
        "-ac",
        str(channels),
        "-ar",
        str(sample_rate),
        "pipe:1",
    ]

    aws_proc = subprocess.Popen(
        aws_cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    ffmpeg_proc = subprocess.Popen(
        ffmpeg_cmd,
        stdin=aws_proc.stdout,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    # Allow ffmpeg to own the read side so the aws process can receive SIGPIPE if ffmpeg exits early.
    if aws_proc.stdout is not None:
        aws_proc.stdout.close()

    return aws_proc, ffmpeg_proc


def read_next_chunk(stream, chunk_bytes):
    parts = []
    bytes_read = 0

    while bytes_read < chunk_bytes:
        piece = stream.read(chunk_bytes - bytes_read)
        if not piece:
            break
        parts.append(piece)
        bytes_read += len(piece)

    if not parts:
        return None

    return b"".join(parts)


def finalize_processes(aws_proc, ffmpeg_proc, s3_uri):
    ffmpeg_stdout = ffmpeg_proc.stdout
    if ffmpeg_stdout is not None:
        ffmpeg_stdout.close()

    ffmpeg_stderr = ffmpeg_proc.stderr.read().decode("utf-8", errors="replace") if ffmpeg_proc.stderr else ""
    aws_stderr = aws_proc.stderr.read().decode("utf-8", errors="replace") if aws_proc.stderr else ""

    ffmpeg_rc = ffmpeg_proc.wait()
    aws_rc = aws_proc.wait()

    if ffmpeg_rc != 0:
        raise RuntimeError(f"ffmpeg failed for {s3_uri}: {ffmpeg_stderr.strip()}")

    if aws_rc != 0:
        raise RuntimeError(f"aws s3 cp failed for {s3_uri}: {aws_stderr.strip()}")


def encode_batch(model, batch_audio, sample_rate, model_half=False, encoder_chunked=False, encoder_overlap=32, encoder_chunk_size=128):
    """Encode a batch of audio chunks with Stable Audio 3 SAME.

    batch_audio shape: [B, C, T], float32 in [-1, 1].
    SAME returns [B, 256, latent_time] for same-l/same-s.
    """
    if model_half:
        batch_audio = batch_audio.to(torch.float16)

    with torch.no_grad():
        latents = model.encode(
            batch_audio,
            sr=sample_rate,
            chunked=encoder_chunked,
            overlap=encoder_overlap,
            chunk_size=encoder_chunk_size,
        )

    return latents.float().cpu().numpy()


def latent_path_for_chunk(output_path, rel_base, chunk_idx):
    return output_path / Path(f"{rel_base}_{chunk_idx:03d}.npy")


def metadata_path_for_chunk(output_path, rel_base, chunk_idx):
    return output_path / Path(f"{rel_base}_{chunk_idx:03d}.json")


CHUNK_STEM_RE = re.compile(r"_(\d+)$")


def row_base_from_sample_stem(stem):
    return CHUNK_STEM_RE.sub("", str(stem))


def row_done_marker(output_path, rel_base):
    return output_path / Path(f"{rel_base}.__done__")


def write_done_marker(marker_path, text="done\n"):
    marker_path.parent.mkdir(parents=True, exist_ok=True)
    marker_path.write_text(text)


def append_decode_error_log(output_path, row_context, error):
    log_path = Path(output_path) / "decode_errors.jsonl"
    payload = {
        "row_index": int(row_context["row_index"]),
        "rel_base": str(row_context["rel_base"]),
        "s3_uri": row_context["s3_uri"],
        "split_name": row_context["split_name"],
        "error": str(error),
    }
    with log_path.open("a") as f:
        f.write(json.dumps(payload, ensure_ascii=True) + "\n")


def cleanup_partial_row_outputs(output_path, row_context, pending_rows):
    row_key = str(row_context["rel_base"])
    pending = pending_rows.pop(row_key, None)
    if pending is not None:
        for meta in pending.get("chunk_metas", []):
            latent_path = meta.get("latent_path")
            if latent_path is not None:
                latent_path = Path(latent_path)
                if latent_path.exists():
                    latent_path.unlink()

    base_glob = f"{row_context['rel_base']}_*"
    split_spool_dir = sample_spool_dir(output_path, row_context["split_name"])
    for path in split_spool_dir.glob(base_glob):
        if path.suffix in {".npy", ".json"} and path.exists():
            path.unlink()


def format_duration(seconds):
    seconds = max(0, int(round(seconds)))
    hours, rem = divmod(seconds, 3600)
    minutes, secs = divmod(rem, 60)
    if hours > 0:
        return f"{hours:d}h{minutes:02d}m{secs:02d}s"
    if minutes > 0:
        return f"{minutes:d}m{secs:02d}s"
    return f"{secs:d}s"


def get_gpu_stats(device):
    if not torch.cuda.is_available():
        return None

    device_str = str(device)
    if not device_str.startswith("cuda"):
        return None

    visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
    device_index = 0
    if ":" in device_str:
        try:
            device_index = int(device_str.split(":", 1)[1])
        except Exception:
            device_index = 0
    elif visible_devices:
        device_index = 0

    try:
        out = subprocess.check_output(
            [
                "nvidia-smi",
                f"--id={device_index}",
                "--query-gpu=utilization.gpu,memory.used,memory.total",
                "--format=csv,noheader,nounits",
            ],
            stderr=subprocess.DEVNULL,
        ).decode("utf-8").strip()
        util, mem_used, mem_total = [x.strip() for x in out.split(",")]
        return {
            "util": int(util),
            "mem_used": int(mem_used),
            "mem_total": int(mem_total),
        }
    except Exception:
        return None


GPU_STATS_CACHE = {}


def load_same_autoencoder(model_name):
    """Load a Stable Audio 3 SAME autoencoder.

    The SA3 docs use aliases like "same-l". Some workflows refer to the
    Hugging Face repo id "stabilityai/SAME-L". Try the requested name first,
    then the corresponding alias/repo fallback for convenience.
    """
    try:
        return AutoencoderModel.from_pretrained(model_name)
    except Exception:
        normalized = str(model_name).strip().lower()
        if normalized in {"stabilityai/same-l", "same-l", "same_l", "samel"}:
            fallback = "same-l" if normalized != "same-l" else "stabilityai/SAME-L"
            return AutoencoderModel.from_pretrained(fallback)
        if normalized in {"stabilityai/same-s", "same-s", "same_s", "sames"}:
            fallback = "same-s" if normalized != "same-s" else "stabilityai/SAME-S"
            return AutoencoderModel.from_pretrained(fallback)
        raise


def format_gpu_stats(device):
    stats = get_gpu_stats(device)
    if stats is None:
        return "GPU=n/a"
    return f"GPU={stats['util']}% VRAM={stats['mem_used']}/{stats['mem_total']}MiB"


def format_gpu_stats_cached(device, min_interval):
    if min_interval <= 0:
        return format_gpu_stats(device)

    now = time.time()
    cache_key = str(device)
    cached = GPU_STATS_CACHE.get(cache_key)
    if cached is not None:
        cached_at, cached_value = cached
        if now - cached_at < min_interval:
            return cached_value

    value = format_gpu_stats(device)
    GPU_STATS_CACHE[cache_key] = (now, value)
    return value


class RuntimeStats:
    def __init__(self):
        self.lock = threading.Lock()
        self.decode_active = 0
        self.decode_batches_emitted = 0
        self.decode_chunks_emitted = 0
        self.gpu_batches_flushed = 0
        self.gpu_chunks_flushed = 0

    def incr_decode_active(self, delta):
        with self.lock:
            self.decode_active += delta

    def add_decode_output(self, num_chunks):
        with self.lock:
            self.decode_batches_emitted += 1
            self.decode_chunks_emitted += int(num_chunks)

    def add_gpu_flush(self, num_chunks):
        with self.lock:
            self.gpu_batches_flushed += 1
            self.gpu_chunks_flushed += int(num_chunks)

    def snapshot(self):
        with self.lock:
            return {
                "decode_active": self.decode_active,
                "decode_batches_emitted": self.decode_batches_emitted,
                "decode_chunks_emitted": self.decode_chunks_emitted,
                "gpu_batches_flushed": self.gpu_batches_flushed,
                "gpu_chunks_flushed": self.gpu_chunks_flushed,
            }


def build_row_context(row, row_index, args, output_path):
    s3_uri = row.get(args.s3_uri_column)
    if not s3_uri:
        raise ValueError(f"Row {row_index} is missing {args.s3_uri_column}")

    rel_base = build_output_base(row, args.sample_id_column, args.s3_uri_column)
    done_marker = row_done_marker(output_path, rel_base)
    split_name = assign_split(
        sample_id=str(rel_base),
        val_ratio=args.val_ratio,
        split_seed=args.split_seed,
        train_split_name=args.train_split_name,
        val_split_name=args.val_split_name,
    )

    duration_sec = row.get("duration_sec")
    if duration_sec is None and row.get("duration_ms") is not None:
        duration_sec = float(row["duration_ms"]) / 1000.0
    if duration_sec is None:
        duration_sec = 0.0

    return {
        "row": row,
        "row_index": row_index,
        "s3_uri": s3_uri,
        "rel_base": rel_base,
        "split_name": split_name,
        "done_marker": done_marker,
        "seconds_total": max(float(duration_sec), args.chunk_seconds),
        "manifest_row": clean_manifest_row(row),
        "base_conditioning_metadata": build_base_conditioning_metadata(row),
        "row_started_at": time.time(),
    }


def finalize_row_metadata(row_context, chunk_metas, num_chunks, output_path, sample_rate):
    if num_chunks == 0:
        raise RuntimeError(f"No audio chunks were decoded from {row_context['s3_uri']}")

    sample_records = []

    for meta in chunk_metas:
        padding_mask = F.interpolate(
            meta["padding_mask"].unsqueeze(0).unsqueeze(1),
            size=meta["latent_length"],
            mode="nearest",
        ).squeeze(0).squeeze(0).int().cpu().numpy().tolist()

        chunk_seconds = meta["padding_mask"].sum().item() / sample_rate
        t_start = meta["seconds_start"] / max(row_context["seconds_total"], 1e-6)
        t_end = min(1.0, (meta["seconds_start"] + chunk_seconds) / max(row_context["seconds_total"], 1e-6))

        metadata = {
            "path": row_context["s3_uri"],
            "relpath": str(row_context["rel_base"]),
            "chunk_idx": int(meta["chunk_idx"]),
            "num_chunks": int(num_chunks),
            "timestamps": [float(t_start), float(t_end)],
            "seconds_start": float(meta["seconds_start"]),
            "seconds_total": float(row_context["seconds_total"]),
            "sample_rate": int(sample_rate),
            "padding_mask": padding_mask,
            "manifest_row": row_context["manifest_row"],
        }
        metadata.update(row_context["base_conditioning_metadata"])
        metadata["normalized_track_position"] = float(t_start)

        base_path = sample_spool_base(
            output_path=output_path,
            split_name=row_context["split_name"],
            rel_base=row_context["rel_base"],
            chunk_idx=meta["chunk_idx"],
        )
        metadata_path = Path(f"{base_path}.json")
        metadata_path.parent.mkdir(parents=True, exist_ok=True)
        with open(metadata_path, "w") as f:
            json.dump(metadata, f)

        npy_path = meta["latent_path"]
        sample_stem = f"{row_context['rel_base']}_{int(meta['chunk_idx']):03d}"
        sample_records.append(
            {
                "row_base": str(row_context["rel_base"]),
                "npy_path": npy_path,
                "json_path": metadata_path,
                "npy_arcname": f"{sample_stem}.npy",
                "json_arcname": f"{sample_stem}.json",
            }
        )

    write_done_marker(row_context["done_marker"])
    return sample_records


def maybe_finalize_completed_rows(pending_rows, output_path, sample_rate):
    finalized = []

    for row_key, pending in list(pending_rows.items()):
        expected_num_chunks = pending.get("expected_num_chunks")
        if expected_num_chunks is None:
            continue

        chunk_metas = pending["chunk_metas"]
        if len(chunk_metas) != expected_num_chunks:
            continue

        sample_records = finalize_row_metadata(
            row_context=pending["row_context"],
            chunk_metas=chunk_metas,
            num_chunks=expected_num_chunks,
            output_path=output_path,
            sample_rate=sample_rate,
        )
        finalized.append((pending["row_context"], sample_records))
        del pending_rows[row_key]

    return finalized


def stream_row_batches(row_context, args, sample_rate, audio_channels, runtime_stats=None):
    chunk_samples = int(round(args.chunk_seconds * sample_rate))
    bytes_per_sample = 4
    chunk_bytes = chunk_samples * audio_channels * bytes_per_sample

    aws_proc, ffmpeg_proc = start_audio_stream(
        row_context["s3_uri"],
        sample_rate=sample_rate,
        channels=audio_channels,
        aws_profile=args.aws_profile,
        request_payer=args.request_payer,
    )

    buffered_audio = []
    buffered_meta = []
    chunk_idx = 0

    try:
        while True:
            raw_chunk = read_next_chunk(ffmpeg_proc.stdout, chunk_bytes)
            if raw_chunk is None:
                break

            audio_np = np.frombuffer(raw_chunk, dtype=np.float32)
            valid_samples = audio_np.size // audio_channels
            if valid_samples == 0:
                break

            if valid_samples < chunk_samples:
                pad = np.zeros(chunk_samples * audio_channels, dtype=np.float32)
                pad[: audio_np.size] = audio_np
                audio_np = pad
            elif valid_samples > chunk_samples:
                audio_np = audio_np[: chunk_samples * audio_channels]
                valid_samples = chunk_samples

            audio = torch.from_numpy(audio_np.reshape(chunk_samples, audio_channels).T.copy()).clamp(-1, 1)
            padding_mask = torch.zeros([chunk_samples], dtype=torch.float32)
            padding_mask[:valid_samples] = 1.0

            buffered_audio.append(audio)
            buffered_meta.append(
                {
                    "chunk_idx": chunk_idx,
                    "seconds_start": float(chunk_idx * args.chunk_seconds),
                    "padding_mask": padding_mask,
                }
            )
            chunk_idx += 1

            if len(buffered_audio) >= args.batch_size:
                if runtime_stats is not None:
                    runtime_stats.add_decode_output(len(buffered_audio))
                yield {
                    "type": "batch",
                    "row_context": row_context,
                    "audio": torch.stack(buffered_audio),
                    "metas": list(buffered_meta),
                }
                buffered_audio.clear()
                buffered_meta.clear()

        if buffered_audio:
            if runtime_stats is not None:
                runtime_stats.add_decode_output(len(buffered_audio))
            yield {
                "type": "batch",
                "row_context": row_context,
                "audio": torch.stack(buffered_audio),
                "metas": list(buffered_meta),
            }

        finalize_processes(aws_proc, ffmpeg_proc, row_context["s3_uri"])
        yield {
            "type": "row_done",
            "row_context": row_context,
            "num_chunks": chunk_idx,
        }

    except Exception:
        try:
            aws_proc.kill()
        except Exception:
            pass
        try:
            ffmpeg_proc.kill()
        except Exception:
            pass
        raise

def decode_worker_loop(row_queue, args, sample_rate, audio_channels, output_path, work_queue, runtime_stats=None):
    try:
        while True:
            item = row_queue.get()
            if item is None:
                break

            row_index, row = item
            row_context = build_row_context(row, row_index, args, output_path)

            if args.skip_existing and row_context["done_marker"].exists():
                work_queue.put(
                    {
                        "type": "skip",
                        "row_context": row_context,
                    }
                )
                continue

            if runtime_stats is not None:
                runtime_stats.incr_decode_active(1)
            try:
                try:
                    for item in stream_row_batches(
                        row_context=row_context,
                        args=args,
                        sample_rate=sample_rate,
                        audio_channels=audio_channels,
                        runtime_stats=runtime_stats,
                    ):
                        work_queue.put(item)
                except Exception as exc:
                    if args.on_decode_error == "skip":
                        work_queue.put(
                            {
                                "type": "decode_error",
                                "row_context": row_context,
                                "error": repr(exc),
                            }
                        )
                        continue
                    raise
            finally:
                if runtime_stats is not None:
                    runtime_stats.incr_decode_active(-1)
    except Exception as exc:
        work_queue.put(
            {
                "type": "error",
                "error": repr(exc),
            }
        )
    finally:
        work_queue.put({"type": "worker_done"})


def enqueue_rows(sharded_rows, row_queue, num_workers):
    for row_item in sharded_rows:
        row_queue.put(row_item)

    for _ in range(num_workers):
        row_queue.put(None)


def parse_device_list(args):
    raw_devices = args.devices
    if raw_devices is None:
        if torch.cuda.is_available():
            devices = [f"cuda:{idx}" for idx in range(torch.cuda.device_count())]
            return devices or ["cuda"]
        return [args.device]

    if raw_devices.strip().lower() == "auto":
        if torch.cuda.is_available():
            devices = [f"cuda:{idx}" for idx in range(torch.cuda.device_count())]
            return devices or ["cuda"]
        return ["cpu"]

    devices = [device.strip() for device in raw_devices.split(",") if device.strip()]
    return devices or [args.device]


def format_all_gpu_stats(devices):
    return " | ".join(f"{device} {format_gpu_stats(device)}" for device in devices)


def format_all_gpu_stats_cached(devices, min_interval):
    return " | ".join(
        f"{device} {format_gpu_stats_cached(device, min_interval)}"
        for device in devices
    )


def safe_qsize(q):
    try:
        return q.qsize()
    except (AttributeError, NotImplementedError):
        return 0


def maybe_set_cuda_device(device):
    device = torch.device(device)
    if device.type == "cuda":
        torch.cuda.set_device(device)


def flush_gpu_worker_pending(gpu_pending, model, args, output_path, device, result_queue, runtime_stats=None):
    if not gpu_pending:
        return 0

    maybe_set_cuda_device(device)
    with torch.cuda.device(device) if torch.device(device).type == "cuda" else nullcontext():
        batch = torch.stack([item["audio"] for item in gpu_pending]).to(device)
        latents = encode_batch(
            model,
            batch,
            sample_rate=args.sample_rate,
            model_half=args.model_half,
            encoder_chunked=args.encoder_chunked,
            encoder_overlap=args.encoder_overlap,
            encoder_chunk_size=args.encoder_chunk_size,
        )

    for latent, item in zip(latents, gpu_pending):
        row_context = item["row_context"]
        meta = item["meta"]
        latent_path = Path(
            f"{sample_spool_base(output_path, row_context['split_name'], row_context['rel_base'], meta['chunk_idx'])}.npy"
        )
        latent_path.parent.mkdir(parents=True, exist_ok=True)
        with open(latent_path, "wb") as f:
            np.save(f, latent)
        meta["latent_length"] = int(latent.shape[1])
        meta["latent_path"] = latent_path
        result_queue.put(
            {
                "type": "encoded",
                "row_context": row_context,
                "meta": meta,
            }
        )

    count = len(gpu_pending)
    gpu_pending.clear()

    if runtime_stats is not None:
        runtime_stats.add_gpu_flush(count)

    flush_count = getattr(args, "_gpu_flush_count", 0) + 1
    setattr(args, "_gpu_flush_count", flush_count)
    if (
        args.clear_gpu_cache_every > 0
        and flush_count % args.clear_gpu_cache_every == 0
        and torch.device(device).type == "cuda"
    ):
        torch.cuda.empty_cache()
    if args.gc_every > 0 and flush_count % args.gc_every == 0:
        gc.collect()

    return count


def gpu_encode_worker_loop(device, args, output_path, input_queue, result_queue):
    maybe_set_cuda_device(device)
    print(f"[gpu:{device}] loading SAME autoencoder: {args.same_model}")
    model = load_same_autoencoder(args.same_model)
    model = model.to(device)
    if args.model_half:
        model = model.half()
    model.eval()

    gpu_pending = []
    target_gpu_batch_size = args.target_gpu_batch_size or args.batch_size

    try:
        while True:
            try:
                item = input_queue.get(timeout=1.0)
            except queue.Empty:
                flush_gpu_worker_pending(
                    gpu_pending=gpu_pending,
                    model=model,
                    args=args,
                    output_path=output_path,
                    device=device,
                    result_queue=result_queue,
                    runtime_stats=None,
                )
                continue

            if item is None:
                flush_gpu_worker_pending(
                    gpu_pending=gpu_pending,
                    model=model,
                    args=args,
                    output_path=output_path,
                    device=device,
                    result_queue=result_queue,
                    runtime_stats=None,
                )
                break

            gpu_pending.append(item)
            if len(gpu_pending) >= target_gpu_batch_size:
                flush_gpu_worker_pending(
                    gpu_pending=gpu_pending,
                    model=model,
                    args=args,
                    output_path=output_path,
                    device=device,
                    result_queue=result_queue,
                    runtime_stats=None,
                )
    except Exception as exc:
        result_queue.put(
            {
                "type": "error",
                "error": repr(exc),
                "device": device,
            }
        )
    finally:
        result_queue.put({"type": "gpu_worker_done", "device": device})


def parse_args():
    parser = argparse.ArgumentParser(
        description="Stream audio files from an S3 parquet manifest, chunk them, and pre-encode with the Stable Audio 3 SAME autoencoder."
    )
    parser.add_argument("--manifest-path", type=str, required=True, help="Path to the parquet manifest")
    parser.add_argument("--output-path", type=str, required=True, help="Directory to write .npy/.json outputs")
    parser.add_argument("--same-model", type=str, default="same-l", help="SAME autoencoder variant or HF/local model id. Default: same-l")
    parser.add_argument("--sample-rate", type=int, default=44100, help="Audio sample rate to stream from ffmpeg for SAME. Default: 44100")
    parser.add_argument("--audio-channels", type=int, default=2, help="Audio channels to stream from ffmpeg for SAME. Default: 2 stereo")
    parser.add_argument("--sample-id-column", type=str, default="sample_id", help="Manifest column used for stable output naming")
    parser.add_argument("--s3-uri-column", type=str, default="audio_s3_uri", help="Manifest column containing the source S3 URI")
    parser.add_argument("--chunk-seconds", type=float, default=60.0, help="Chunk duration in seconds")
    parser.add_argument("--val-ratio", type=float, default=0.10, help="Deterministic validation split ratio applied at the source-row level")
    parser.add_argument("--split-seed", type=int, default=0, help="Seed used for deterministic train/val split assignment")
    parser.add_argument("--train-split-name", type=str, default="train", help="Name for the training split")
    parser.add_argument("--val-split-name", type=str, default="val", help="Name for the validation split")
    parser.add_argument("--batch-size", type=int, default=4, help="Number of chunks to encode at once")
    parser.add_argument("--aws-profile", type=str, default=None, help="Optional AWS CLI profile to use")
    parser.add_argument("--request-payer", action="store_true", help="Pass --request-payer requester to aws s3 cp")
    parser.add_argument("--num-shards", type=int, default=1, help="Split manifest rows across N workers/processes")
    parser.add_argument("--shard-index", type=int, default=0, help="Current shard index in [0, num_shards)")
    parser.add_argument("--start-row", type=int, default=0, help="Skip manifest rows before this absolute row index")
    parser.add_argument("--limit", type=int, default=None, help="Optional maximum number of rows to process after sharding")
    parser.add_argument("--skip-existing", action="store_true", help="Skip rows with an existing completion marker")
    parser.add_argument("--on-decode-error", choices=("skip", "fail"), default="skip", help="What to do when aws/ffmpeg cannot decode a source object")
    parser.add_argument("--model-half", action="store_true", help="Run the autoencoder in float16")
    parser.add_argument("--encoder-chunked", action="store_true", help="Use the model's internal chunked encoder for each 60s chunk")
    parser.add_argument("--encoder-overlap", type=int, default=32, help="Overlap, measured in latents, for internal chunked encoding")
    parser.add_argument("--encoder-chunk-size", type=int, default=128, help="Chunk size, measured in latents, for internal chunked encoding")
    parser.add_argument("--target-gpu-batch-size", type=int, default=None, help="Aggregate chunks across rows up to this size before each encode step")
    parser.add_argument("--decode-workers", type=int, default=4, help="Number of concurrent S3/ffmpeg decode workers")
    parser.add_argument("--prefetch-batches", type=int, default=2, help="Number of decoded CPU batches to buffer ahead of the GPU encoder")
    parser.add_argument("--samples-per-shard", type=int, default=5000, help="Number of finalized chunk samples per WebDataset tar shard")
    parser.add_argument("--s3-prefix", type=str, default=None, help="Optional S3 destination prefix for uploaded shards, for example s3://bucket/path")
    parser.add_argument("--state-s3-uri", type=str, default=None, help="Optional S3 URI for persisted shard/upload state; defaults to <s3-prefix>/_state/shard_state.json when --s3-prefix is set")
    parser.add_argument("--shard-name-prefix", type=str, default="", help="Optional prefix for generated shard tar filenames, useful when multiple workers upload to the same S3 prefix")
    parser.add_argument("--keep-local-shards", action="store_true", help="Keep local tar shards after upload")
    parser.add_argument("--keep-local-samples", action="store_true", help="Keep local per-sample .npy/.json spool files after they are packed into a shard")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Fallback torch device for encoding when no CUDA devices are available")
    parser.add_argument("--devices", type=str, default=None, help="Comma-separated torch devices for multi-GPU encoding, for example cuda:0,cuda:1,cuda:2. Defaults to all visible CUDA devices; use 'auto' to request the same behavior explicitly.")
    parser.add_argument("--gpu-stats-interval", type=float, default=10.0, help="Minimum seconds between nvidia-smi refreshes in progress output. Set to 0 to refresh every progress line.")
    parser.add_argument("--clear-gpu-cache-every", type=int, default=0, help="Call torch.cuda.empty_cache() every N GPU encode flushes. Default 0 disables periodic cache clearing.")
    parser.add_argument("--gc-every", type=int, default=0, help="Run Python gc.collect() every N GPU encode flushes. Default 0 disables periodic explicit GC.")
    return parser.parse_args()


def main():
    args = parse_args()

    if args.samples_per_shard <= 0:
        raise ValueError("--samples-per-shard must be greater than 0")
    if not 0.0 <= args.val_ratio <= 1.0:
        raise ValueError("--val-ratio must be between 0 and 1")
    if args.keep_local_shards and args.s3_prefix is None:
        print("--keep-local-shards set without --s3-prefix; local shards will simply be retained")
    if args.state_s3_uri is None and args.s3_prefix is not None:
        args.state_s3_uri = f"{args.s3_prefix.rstrip('/')}/_state/shard_state.json"

    devices = parse_device_list(args)
    single_device_inline = len(devices) == 1
    inline_model = None
    if single_device_inline:
        device = devices[0]
        print(f"[gpu:{device}] loading model inline")
        maybe_set_cuda_device(device)
        inline_model = load_same_autoencoder(args.same_model)
        inline_model = inline_model.to(device)
        if args.model_half:
            inline_model = inline_model.half()
        inline_model.eval()

    sample_rate = int(args.sample_rate)
    audio_channels = int(args.audio_channels)

    output_path = Path(args.output_path)
    output_path.mkdir(parents=True, exist_ok=True)
    split_names = (args.train_split_name, args.val_split_name)
    shard_manager = ShardWriterManager(
        output_path=output_path,
        split_names=split_names,
        samples_per_shard=args.samples_per_shard,
        s3_prefix=args.s3_prefix,
        aws_profile=args.aws_profile,
        request_payer=args.request_payer,
        keep_local_shards=args.keep_local_shards,
        keep_local_samples=args.keep_local_samples,
        state_s3_uri=args.state_s3_uri,
        shard_name_prefix=args.shard_name_prefix,
    )
    recovered_local_samples = shard_manager.recover_local_samples()

    rows = read_manifest(args.manifest_path)
    sharded_rows = shard_rows(
        rows,
        num_shards=args.num_shards,
        shard_index=args.shard_index,
        start_row=args.start_row,
        limit=args.limit,
    )

    pre_skipped = 0
    if args.skip_existing:
        sharded_rows, pre_skipped = filter_existing_rows(
            sharded_rows,
            output_path=output_path,
            sample_id_column=args.sample_id_column,
            s3_uri_column=args.s3_uri_column,
            completed_rows=shard_manager.state.get("completed_rows", []),
        )

    details = {
        "manifest_path": args.manifest_path,
        "same_model": args.same_model,
        "sample_rate": sample_rate,
        "audio_channels": audio_channels,
        "devices": devices,
        "args": vars(args),
        "rows_in_manifest": len(rows),
        "rows_in_this_shard": len(sharded_rows),
        "rows_pre_skipped": pre_skipped,
        "split_names": list(split_names),
        "recovered_local_samples": recovered_local_samples,
    }
    (output_path / "details.json").write_text(json.dumps(details, indent=2))

    print(f"Processing {len(sharded_rows)} manifest rows on shard {args.shard_index}/{args.num_shards}")
    if pre_skipped:
        print(f"Pre-skipped {pre_skipped} rows with existing done markers")

    started_at = time.time()
    completed = 0
    total_rows = len(sharded_rows)
    pending_rows = {}
    target_gpu_batch_size = args.target_gpu_batch_size or args.batch_size
    runtime_stats = RuntimeStats()
    work_queue = queue.Queue(maxsize=max(1, args.prefetch_batches * max(1, args.decode_workers)))
    row_queue = queue.Queue(maxsize=max(1, args.decode_workers * 2))
    gpu_context = None if single_device_inline else mp.get_context("spawn")
    gpu_result_queue = queue.Queue() if single_device_inline else gpu_context.Queue()
    gpu_queues = [] if single_device_inline else [
        gpu_context.Queue(maxsize=max(1, args.prefetch_batches * max(1, target_gpu_batch_size)))
        for _ in devices
    ]
    gpu_pending = []
    workers = []
    gpu_workers = []

    for _ in range(max(1, args.decode_workers)):
        worker = threading.Thread(
            target=decode_worker_loop,
            args=(row_queue, args, sample_rate, audio_channels, output_path, work_queue, runtime_stats),
            daemon=True,
        )
        worker.start()
        workers.append(worker)

    if not single_device_inline:
        for gpu_queue, device in zip(gpu_queues, devices):
            worker = gpu_context.Process(
                target=gpu_encode_worker_loop,
                args=(device, args, output_path, gpu_queue, gpu_result_queue),
                daemon=True,
            )
            worker.start()
            gpu_workers.append(worker)

    enqueue_thread = threading.Thread(
        target=enqueue_rows,
        args=(sharded_rows, row_queue, len(workers)),
        daemon=True,
    )
    enqueue_thread.start()

    finished_workers = 0
    finished_gpu_workers = 0
    gpu_stop_sent = False
    gpu_dispatch_index = 0
    reported_dead_gpu_workers = set()

    def pending_gpu_count():
        if single_device_inline:
            return len(gpu_pending)
        return sum(safe_qsize(gpu_queue) for gpu_queue in gpu_queues)

    def flush_inline_gpu_pending():
        if not single_device_inline:
            return 0
        return flush_gpu_worker_pending(
            gpu_pending=gpu_pending,
            model=inline_model,
            args=args,
            output_path=output_path,
            device=devices[0],
            result_queue=gpu_result_queue,
            runtime_stats=None,
        )

    def finalize_ready_rows():
        for row_context in maybe_finalize_completed_rows(
            pending_rows=pending_rows,
            output_path=output_path,
            sample_rate=sample_rate,
        ):
            shard_manager.add_samples(row_context[0]["split_name"], row_context[1])
            emit_progress(row_context[0])

    def handle_gpu_result(item):
        nonlocal finished_gpu_workers

        if item["type"] == "encoded":
            row_context = item["row_context"]
            row_key = str(row_context["rel_base"])
            pending = pending_rows.setdefault(
                row_key,
                {
                    "row_context": row_context,
                    "chunk_metas": [],
                },
            )
            pending["row_context"] = row_context
            pending["chunk_metas"].append(item["meta"])
            runtime_stats.add_gpu_flush(1)
            finalize_ready_rows()
            return

        if item["type"] == "gpu_worker_done":
            finished_gpu_workers += 1
            return

        if item["type"] == "error":
            raise RuntimeError(f"GPU producer failed on {item.get('device')}: {item['error']}")

    def drain_gpu_results():
        drained = 0
        while True:
            try:
                item = gpu_result_queue.get_nowait()
            except queue.Empty:
                break
            handle_gpu_result(item)
            drained += 1
        return drained

    def check_gpu_worker_health():
        for idx, worker in enumerate(gpu_workers):
            if idx in reported_dead_gpu_workers:
                continue
            if worker.exitcode is None:
                continue
            if worker.exitcode != 0:
                reported_dead_gpu_workers.add(idx)
                raise RuntimeError(
                    f"GPU worker for {devices[idx]} exited with status {worker.exitcode}"
                )

    def emit_progress(row_context, skipped=False):
        nonlocal completed
        completed += 1
        elapsed = time.time() - started_at
        avg_sec_per_row = elapsed / max(completed, 1)
        remaining = max(total_rows - completed, 0)
        eta_seconds = remaining * avg_sec_per_row
        rows_per_min = 60.0 / avg_sec_per_row if avg_sec_per_row > 0 else 0.0
        pct = (100.0 * completed / total_rows) if total_rows else 100.0
        stats = runtime_stats.snapshot()
        chunks_per_sec = stats["gpu_chunks_flushed"] / elapsed if elapsed > 0 else 0.0

        if skipped:
            print(f"Skipping row {row_context['row_index']}: already complete -> {row_context['rel_base']}")
            print(
                f"Progress: {completed}/{total_rows} rows ({pct:.2f}%) | "
                f"throughput={rows_per_min:.2f} rows/min | ETA={format_duration(eta_seconds)} | "
                f"pending_gpu={pending_gpu_count()} row_q={row_queue.qsize()} work_q={work_queue.qsize()} "
                f"decode_active={stats['decode_active']}/{max(1, args.decode_workers)} "
                f"decoded_chunks={stats['decode_chunks_emitted']} gpu_chunks={stats['gpu_chunks_flushed']} "
                f"chunks/s={chunks_per_sec:.2f} | {format_all_gpu_stats_cached(devices, args.gpu_stats_interval)}"
            )
            return

        row_elapsed = time.time() - row_context["row_started_at"]
        expected_num_chunks = row_context.get("expected_num_chunks", "?")
        print(f"Finished row {row_context['row_index']}: {row_context['s3_uri']} -> {expected_num_chunks} chunks")
        print(
            f"Progress: {completed}/{total_rows} rows ({pct:.2f}%) | "
            f"last={row_elapsed:.1f}s avg={avg_sec_per_row:.1f}s | "
            f"throughput={rows_per_min:.2f} rows/min | ETA={format_duration(eta_seconds)} | "
            f"pending_gpu={pending_gpu_count()} row_q={row_queue.qsize()} work_q={work_queue.qsize()} "
            f"decode_active={stats['decode_active']}/{max(1, args.decode_workers)} "
            f"decoded_chunks={stats['decode_chunks_emitted']} gpu_chunks={stats['gpu_chunks_flushed']} "
            f"chunks/s={chunks_per_sec:.2f} | {format_all_gpu_stats_cached(devices, args.gpu_stats_interval)}"
        )

    while True:
        if single_device_inline and len(gpu_pending) >= target_gpu_batch_size:
            flush_inline_gpu_pending()

        drain_gpu_results()
        check_gpu_worker_health()

        if single_device_inline and finished_workers == len(workers):
            flush_inline_gpu_pending()
            drain_gpu_results()
            shard_manager.flush_all()

            if pending_rows:
                sample_rows = list(pending_rows.values())[:10]
                sample_ids = [str(x["row_context"]["rel_base"]) for x in sample_rows]
                raise RuntimeError(
                    "Shutting down with unresolved pending rows still in memory. "
                    f"pending_rows={len(pending_rows)} sample_ids={sample_ids}"
                )
            break

        if not single_device_inline and finished_workers == len(workers) and not gpu_stop_sent:
            for gpu_queue in gpu_queues:
                gpu_queue.put(None)
            gpu_stop_sent = True

        if not single_device_inline and gpu_stop_sent and finished_gpu_workers == len(gpu_workers):
            drain_gpu_results()
            shard_manager.flush_all()

            if pending_rows:
                sample_rows = list(pending_rows.values())[:10]
                sample_ids = [str(x["row_context"]["rel_base"]) for x in sample_rows]
                raise RuntimeError(
                    "Shutting down with unresolved pending rows still in memory. "
                    f"pending_rows={len(pending_rows)} sample_ids={sample_ids}"
                )
            break

        try:
            item = work_queue.get(timeout=0.2)
        except queue.Empty:
            continue

        if item["type"] == "batch":
            row_context = item["row_context"]
            for audio, meta in zip(item["audio"], item["metas"]):
                gpu_item = {
                    "row_context": row_context,
                    "audio": audio,
                    "meta": meta,
                }
                if single_device_inline:
                    gpu_pending.append(gpu_item)
                    if len(gpu_pending) >= target_gpu_batch_size:
                        flush_inline_gpu_pending()
                        drain_gpu_results()
                else:
                    gpu_queues[gpu_dispatch_index].put(gpu_item)
                    gpu_dispatch_index = (gpu_dispatch_index + 1) % len(gpu_queues)
            continue

        if item["type"] == "row_done":
            row_context = item["row_context"]
            row_key = str(row_context["rel_base"])
            pending = pending_rows.setdefault(
                row_key,
                {
                    "row_context": row_context,
                    "chunk_metas": [],
                },
            )
            pending["row_context"] = row_context
            pending["expected_num_chunks"] = item["num_chunks"]
            row_context["expected_num_chunks"] = item["num_chunks"]
            finalize_ready_rows()
            continue

        if item["type"] == "skip":
            row_context = item["row_context"]
            emit_progress(row_context, skipped=True)
            continue

        if item["type"] == "decode_error":
            row_context = item["row_context"]
            cleanup_partial_row_outputs(output_path, row_context, pending_rows)
            append_decode_error_log(output_path, row_context, item["error"])
            write_done_marker(row_context["done_marker"], text="decode_error\n")
            print(
                f"Skipping row {row_context['row_index']}: decode failed -> "
                f"{row_context['s3_uri']} | {item['error']}"
            )
            emit_progress(row_context, skipped=True)
            continue

        if item["type"] == "error":
            raise RuntimeError(f"Producer failed: {item['error']}")

        if item["type"] == "worker_done":
            finished_workers += 1
            continue

    for worker in workers:
        worker.join()
    for worker in gpu_workers:
        worker.join()
    enqueue_thread.join()
    shard_manager.flush_all()


if __name__ == "__main__":
    main()
