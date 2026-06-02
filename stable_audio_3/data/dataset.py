import numpy as np
import io
import json
import os
import posixpath
import re
import shlex
import subprocess
import sys
import dill
import random
import time
import torch
import torchaudio

from os import path
from pathlib import Path
from torchaudio import transforms as T
from typing import Optional, Callable, List

from .utils import Stereo, Mono, PhaseFlipper, PadCrop_Normalized_T, VolumeNorm, strip_trailing_silence

AUDIO_KEYS = ("flac", "wav", "mp3", "m4a", "ogg", "opus")
RANDOM_PROMPT_KEYS = ("dense_caption", "vivid_caption", "one_line_caption")
CAPTION_BUCKET_PROBS = (
    ("dense", 0.30),
    ("vivid", 0.375),
    ("one_line", 0.325),
)
DENSE_POLICY_PROBS = (
    ("keep_structured_heading", 0.40),
    ("omit_structured_heading", 0.30),
    ("replace_with_simple_prefix", 0.30),
)
BPM_NUMERIC_ACTIVE_PROB = 0.70


def is_valid_prompt_value(prompt):
    if prompt is None:
        return False

    if isinstance(prompt, str):
        return bool(prompt.strip())

    return True


def has_random_prompt_fields(metadata):
    return all(is_valid_prompt_value(metadata.get(key)) for key in RANDOM_PROMPT_KEYS)


def has_any_random_prompt_field(metadata):
    return any(key in metadata for key in RANDOM_PROMPT_KEYS)


def has_valid_prompt(metadata):
    if has_any_random_prompt_field(metadata):
        return has_random_prompt_fields(metadata)

    return is_valid_prompt_value(metadata.get("prompt"))


def sample_weighted_choice(weighted_items):
    total_weight = sum(weight for _, weight in weighted_items)
    draw = random.random() * total_weight
    cumulative = 0.0

    for item, weight in weighted_items:
        cumulative += weight
        if draw < cumulative:
            return item

    return weighted_items[-1][0]


def strip_structured_heading(caption):
    if not isinstance(caption, str):
        return caption

    lines = caption.splitlines()
    first_content_ix = None

    for ix, line in enumerate(lines):
        if line.strip():
            first_content_ix = ix
            break

    if first_content_ix is None:
        return caption

    genre_ix = first_content_ix
    if not re.match(r"^\s*Genre\s*:", lines[genre_ix], flags=re.IGNORECASE):
        return caption.strip()

    end_ix = genre_ix + 1
    while end_ix < len(lines):
        stripped = lines[end_ix].strip()

        if not stripped:
            end_ix += 1
            break

        if re.match(r"^\s*Sub[- ]?genres?\s*:", lines[end_ix], flags=re.IGNORECASE):
            end_ix += 1
            continue

        break

    return "\n".join(lines[end_ix:]).strip()


def coerce_subgenres(subgenres):
    if subgenres is None:
        return []

    if isinstance(subgenres, str):
        return [item.strip() for item in subgenres.split(",") if item.strip()]

    if isinstance(subgenres, (list, tuple)):
        return [str(item).strip() for item in subgenres if str(item).strip()]

    return []


def make_simple_genre_prefix(metadata):
    main = metadata.get("genre") or metadata.get("primary_genre")
    if not main:
        return ""

    main = str(main).strip()
    subgenres = [
        subgenre
        for subgenre in coerce_subgenres(metadata.get("subgenres"))
        if subgenre.lower() != main.lower()
    ][:3]

    if subgenres:
        return f"{main}: {', '.join(subgenres)}."

    return f"{main}."


def is_truthy_metadata_value(value):
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "t", "yes", "y"}

    return bool(value)


def build_ttm_condition(metadata):
    bucket = sample_weighted_choice(CAPTION_BUCKET_PROBS)

    if bucket == "dense":
        dense_caption = metadata["dense_caption"]
        dense_prose = strip_structured_heading(dense_caption)
        policy = sample_weighted_choice(DENSE_POLICY_PROBS)

        if policy == "keep_structured_heading":
            text = dense_caption
        elif policy == "omit_structured_heading":
            text = dense_prose
        else:
            prefix = make_simple_genre_prefix(metadata)
            text = f"{prefix} {dense_prose}".strip() if prefix else dense_prose
    elif bucket == "vivid":
        text = metadata["vivid_caption"]
    else:
        text = metadata["one_line_caption"]

    tempo = None

    if is_truthy_metadata_value(metadata.get("bpm_valid")):
        try:
            bpm = round(float(metadata["bpm"]))
        except (KeyError, TypeError, ValueError):
            bpm = None

        if bpm is not None:
            if bucket in {"dense", "vivid"}:
                text = f"{bpm} BPM. {text}"
            else:
                text = f"{bpm} BPM {text}"

            if random.random() < BPM_NUMERIC_ACTIVE_PROB:
                tempo = bpm

    return {
        **metadata,
        "prompt": text,
        "tempo": tempo,
    }


def maybe_build_ttm_condition(metadata):
    if has_random_prompt_fields(metadata):
        return build_ttm_condition(metadata)
    return metadata

DEFAULT_S3_STREAMING_CONFIG = {
    "cli_connect_timeout_sec": 30,
    "cli_read_timeout_sec": 120,
    "ls_timeout_sec": 300,
    "stream_timeout_sec": 900,
    "stream_idle_timeout_sec": 300,
    "max_attempts": 3,
    "retry_mode": "standard",
}


def _require_webdataset():
    try:
        import webdataset as wds
    except ImportError as exc:
        raise ImportError(
            "S3 WebDataset training requires the optional 'webdataset' package. "
            "Install the training extras or add webdataset to your environment."
        ) from exc
    return wds


def normalize_s3_streaming_config(overrides=None):
    config = dict(DEFAULT_S3_STREAMING_CONFIG)
    if overrides:
        config.update({key: value for key, value in overrides.items() if value is not None})

    for key in (
        "cli_connect_timeout_sec",
        "cli_read_timeout_sec",
        "ls_timeout_sec",
        "stream_timeout_sec",
        "stream_idle_timeout_sec",
        "max_attempts",
    ):
        try:
            config[key] = int(config[key])
        except (TypeError, ValueError) as exc:
            raise ValueError(f"s3_streaming.{key} must be an integer") from exc
        if config[key] <= 0:
            raise ValueError(f"s3_streaming.{key} must be greater than 0")

    retry_mode = str(config["retry_mode"]).strip()
    if not retry_mode:
        raise ValueError("s3_streaming.retry_mode must be a non-empty string")
    config["retry_mode"] = retry_mode
    return config


def build_aws_cli_base_cmd(s3_streaming_config, profile=None):
    config = normalize_s3_streaming_config(s3_streaming_config)
    cmd = [
        "aws",
        "--cli-connect-timeout",
        str(config["cli_connect_timeout_sec"]),
        "--cli-read-timeout",
        str(config["cli_read_timeout_sec"]),
    ]
    if profile is not None:
        cmd.extend(["--profile", profile])
    return cmd


def build_aws_cli_env(s3_streaming_config):
    config = normalize_s3_streaming_config(s3_streaming_config)
    env = os.environ.copy()
    env["AWS_MAX_ATTEMPTS"] = str(config["max_attempts"])
    env["AWS_RETRY_MODE"] = config["retry_mode"]
    return env


def build_s3_pipe_request(s3_path, profile=None, s3_streaming_config=None):
    config = normalize_s3_streaming_config(s3_streaming_config)
    s3_pipe_script = Path(__file__).with_name("s3_pipe.py")
    cmd = [
        shlex.quote(sys.executable),
        shlex.quote(str(s3_pipe_script)),
        "--s3-path",
        shlex.quote(s3_path),
        "--cli-connect-timeout-sec",
        str(config["cli_connect_timeout_sec"]),
        "--cli-read-timeout-sec",
        str(config["cli_read_timeout_sec"]),
        "--stream-timeout-sec",
        str(config["stream_timeout_sec"]),
        "--stream-idle-timeout-sec",
        str(config["stream_idle_timeout_sec"]),
        "--max-attempts",
        str(config["max_attempts"]),
        "--retry-mode",
        shlex.quote(config["retry_mode"]),
    ]
    if profile is not None:
        cmd.extend(["--profile", shlex.quote(profile)])
    return f"pipe:{' '.join(cmd)}"

# fast_scandir implementation by Scott Hawley originally in https://github.com/zqevans/audio-diffusion/blob/main/dataset/dataset.py

def fast_scandir(
    dir:str,  # top-level directory at which to begin scanning
    ext:list,  # list of allowed file extensions,
    #max_size = 1 * 1000 * 1000 * 1000 # Only files < 1 GB
    ):
    "very fast `glob` alternative. from https://stackoverflow.com/a/59803793/4259243"
    subfolders, files = [], []
    ext = ['.'+x if x[0]!='.' else x for x in ext]  # add starting period to extensions if needed
    try: # hope to avoid 'permission denied' by this try
        for f in os.scandir(dir):
            try: # 'hope to avoid too many levels of symbolic links' error
                if f.is_dir():
                    subfolders.append(f.path)
                elif f.is_file():
                    file_ext = os.path.splitext(f.name)[1].lower()
                    is_hidden = os.path.basename(f.path).startswith(".")

                    if file_ext in ext and not is_hidden:
                        files.append(f.path)
            except:
                pass 
    except:
        pass

    for dir in list(subfolders):
        sf, f = fast_scandir(dir, ext)
        subfolders.extend(sf)
        files.extend(f)
    return subfolders, files

def keyword_scandir(
    dir: str,  # top-level directory at which to begin scanning
    ext: list,  # list of allowed file extensions
    keywords: list,  # list of keywords to search for in the file name
):
    "very fast `glob` alternative. from https://stackoverflow.com/a/59803793/4259243"
    subfolders, files = [], []
    # make keywords case insensitive
    keywords = [keyword.lower() for keyword in keywords]
    # add starting period to extensions if needed
    ext = ['.'+x if x[0] != '.' else x for x in ext]
    banned_words = ["paxheader", "__macosx"]
    try:  # hope to avoid 'permission denied' by this try
        for f in os.scandir(dir):
            try:  # 'hope to avoid too many levels of symbolic links' error
                if f.is_dir():
                    subfolders.append(f.path)
                elif f.is_file():
                    is_hidden = f.name.split("/")[-1][0] == '.'
                    has_ext = os.path.splitext(f.name)[1].lower() in ext
                    name_lower = f.name.lower()
                    has_keyword = any(
                        [keyword in name_lower for keyword in keywords])
                    has_banned = any(
                        [banned_word in name_lower for banned_word in banned_words])
                    if has_ext and has_keyword and not has_banned and not is_hidden and not os.path.basename(f.path).startswith("._"):
                        files.append(f.path)
            except:
                pass
    except:
        pass

    for dir in list(subfolders):
        sf, f = keyword_scandir(dir, ext, keywords)
        subfolders.extend(sf)
        files.extend(f)
    return subfolders, files

def get_audio_filenames(
    paths: list,  # directories in which to search
    keywords=None,
    exts=['.wav', '.mp3', '.flac', '.ogg', '.aif', '.opus'],
    filelist_path=None
):
    "recursively get a list of audio filenames"
    filenames = []
    if type(paths) is str:
        paths = [paths]
    for path in paths:               # get a list of relevant filenames

        if filelist_path is None:
            # Check for filelist.txt at the root of the directory
            filelist_path = os.path.join(path, "filelist.txt")
            
        if os.path.exists(filelist_path):
            with open(filelist_path, "r") as f:
                files = f.readlines()
                files = [os.path.join(path, file.strip()) for file in files]
                filenames.extend(files)
            continue

        if keywords is not None:
            subfolders, files = keyword_scandir(path, exts, keywords)
        else:
            subfolders, files = fast_scandir(path, exts)
        filenames.extend(files)
    return filenames

def get_latent_filenames(
    paths,  # directories in which to search
    extension='npy',
    filelist_path=None
):
    "recursively get a list of pre-encoded filenames"
    filenames = []
    if type(paths) is str:
        paths = [paths]
    for path in paths:               # get a list of relevant filenames

        if filelist_path is None:
            # Check for filelist.txt at the root of the directory
            filelist_path = os.path.join(path, "filelist.txt")
        
        if os.path.exists(filelist_path):
            with open(filelist_path, "r") as f:
                files = f.readlines()
                files = [os.path.join(path, file.strip()) for file in files]
                filenames.extend(files)
            continue

        _, files = fast_scandir(path, [extension])
        filenames.extend(files)

    # Filter out silence.npy (used for silence latent padding, not a data sample)
    filenames = [f for f in filenames if os.path.basename(f) != "silence.npy"]

    # Add metadata paths
    filenames = [(filename, filename.replace(f".{extension}", ".json")) for filename in filenames]

    return filenames

class LocalDatasetConfig:
    def __init__(
        self,
        id: str,
        path: str,
        keywords: Optional[List[str]]=None,
        custom_metadata_fn: Optional[Callable[[str], str]] = None,
        filelist_path = None,
        weight: float = 1.0,
    ):
        self.id = id
        self.path = path
        self.custom_metadata_fn = custom_metadata_fn
        self.keywords = keywords
        self.filelist_path = filelist_path
        self.weight = weight

class LatentDatasetConfig(LocalDatasetConfig):
    def __init__(
        self,
        latent_extension: str = "npy",
        filelist_path = None,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.latent_extension = latent_extension
        self.filelist_path = filelist_path
        # weight is inherited from LocalDatasetConfig via **kwargs


class LocalWebDatasetConfig:
    def __init__(
        self,
        id: str,
        path: str,
        custom_metadata_fn: Optional[Callable[[dict, torch.Tensor], dict]] = None,
    ):
        self.id = id
        self.path = path
        self.custom_metadata_fn = custom_metadata_fn
        self.urls = []

    def load_data_urls(self):
        _, self.urls = fast_scandir(self.path, ["tar"])
        return self.urls


class S3DatasetConfig:
    def __init__(
        self,
        id: str,
        s3_path: str,
        custom_metadata_fn: Optional[Callable[[dict, torch.Tensor], dict]] = None,
        profile: Optional[str] = None,
        s3_streaming_config: Optional[dict] = None,
    ):
        self.id = id
        self.path = s3_path
        self.custom_metadata_fn = custom_metadata_fn
        self.profile = profile
        self.s3_streaming_config = normalize_s3_streaming_config(s3_streaming_config)
        self.urls = []

    def load_data_urls(self):
        self.urls = get_all_s3_urls(
            names=[self.path],
            s3_url_prefix=None,
            recursive=True,
            profiles={self.path: self.profile} if self.profile else {},
            s3_streaming_configs={self.path: self.s3_streaming_config},
        )
        return self.urls


class SampleDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        configs,
        sample_size=65536,
        sample_rate=48000,
        random_crop=True,
        force_channels="stereo",
        volume_norm=False,
        volume_norm_param=(-16, 2),
        strip_silence=False,
        pad=True,
    ):
        super().__init__()
        self.filenames = []
        self.sample_weights = []

        self.augs = torch.nn.Sequential(
            PhaseFlipper(),
            #nn.Identity()
        )


        self.root_paths = []

        self.pad_crop = PadCrop_Normalized_T(sample_size, sample_rate, randomize=random_crop, pad=pad)
        self.strip_silence = strip_silence

        self.force_channels = force_channels

        self.encoding = torch.nn.Sequential(
            Stereo() if self.force_channels == "stereo" else torch.nn.Identity(),
            Mono() if self.force_channels == "mono" else torch.nn.Identity()
        )

        self.sr = sample_rate

        self.volume_norm = VolumeNorm(volume_norm_param, self.sr) if volume_norm else torch.nn.Identity()

        self.custom_metadata_fns = {}

        for config in configs:
            self.root_paths.append(config.path)
            new_files = get_audio_filenames(config.path, config.keywords, filelist_path=config.filelist_path)
            self.filenames.extend(new_files)
            self.sample_weights.extend([config.weight] * len(new_files))
            if config.custom_metadata_fn is not None:
                self.custom_metadata_fns[config.path] = dill.dumps(config.custom_metadata_fn)

        print(f'Found {len(self.filenames)} files')

    def load_file(self, filename):
        ext = filename.split(".")[-1]

        audio, in_sr = torchaudio.load(filename, format=ext)

        if in_sr != self.sr:
            resample_tf = T.Resample(in_sr, self.sr)
            audio = resample_tf(audio)

        return audio

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        audio_filename = self.filenames[idx]
        try:
            start_time = time.time()
            audio = self.load_file(audio_filename)

            audio = self.volume_norm(audio)

            if self.strip_silence:
                audio = strip_trailing_silence(audio, self.sr)

            audio, t_start, t_end, seconds_start, seconds_total, padding_mask = self.pad_crop(audio)

            # Check for silence
            if is_silence(audio):
                return self[random.randrange(len(self))]

            # Run augmentations on this sample (including random crop)
            if self.augs is not None:
                audio = self.augs(audio)

            audio = audio.clamp(-1, 1)

            # Encode the file to assist in prediction
            if self.encoding is not None:
                audio = self.encoding(audio)

            info = {}

            info["path"] = audio_filename

            for root_path in self.root_paths:
                if root_path in audio_filename:
                    info["relpath"] = path.relpath(audio_filename, root_path)

            info["timestamps"] = (t_start, t_end)
            info["seconds_start"] = seconds_start
            info["seconds_total"] = seconds_total
            info["padding_mask"] = [padding_mask]
            info["sample_rate"] = self.sr

            end_time = time.time()

            info["load_time"] = end_time - start_time

            for custom_md_path in self.custom_metadata_fns.keys():
                if custom_md_path in audio_filename:
                    custom_metadata_fn = dill.loads(self.custom_metadata_fns[custom_md_path])
                    custom_metadata = custom_metadata_fn(info, audio)
                    info.update(custom_metadata)

                if "__reject__" in info and info["__reject__"]:
                    return self[random.randrange(len(self))]

                # Provide audio inputs as their own dictionary to be merged into info, each audio element will be normalized in the same way as the main audio
                if "__audio__" in info:
                    for audio_key, audio_value in info["__audio__"].items():
                        # Process the audio_value tensor, which should be a torch tensor
                        audio_value, _, _, _, _, _ = self.pad_crop(audio_value)
                        audio_value = audio_value.clamp(-1, 1)
                        if self.encoding is not None:
                            audio_value = self.encoding(audio_value)
                        info[audio_key] = audio_value
                
                    del info["__audio__"]

            info = maybe_build_ttm_condition(info)

            return (audio, info)
        except Exception as e:
            print(f'Couldn\'t load file {audio_filename}: {e}')
            return self[random.randrange(len(self))]


class PreEncodedDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        configs: List[LatentDatasetConfig],
        latent_crop_length=None,
        min_length_sec=None,
        max_length_sec=None,
        random_crop=False,
        tokenizers: Optional[dict] = None,
    ):
        super().__init__()
        self.filenames = []
        self.sample_weights = []

        self.custom_metadata_fns = {}

        self.silence_latents = {}

        for config in configs:
            new_files = get_latent_filenames(config.path, config.latent_extension, config.filelist_path)
            self.filenames.extend(new_files)
            self.sample_weights.extend([config.weight] * len(new_files))
            if config.custom_metadata_fn is not None:
                self.custom_metadata_fns[config.path] = dill.dumps(config.custom_metadata_fn)

            # Load silence latent if available (for variable-length padding)
            paths = config.path if isinstance(config.path, list) else [config.path]
            for path in paths:
                silence_path = os.path.join(path, "silence.npy")
                if os.path.exists(silence_path):
                    self.silence_latents[path] = np.load(silence_path).squeeze(0)  # [C, N]
                    print(f'Loaded silence latent from {silence_path}')

        self.latent_crop_length = latent_crop_length
        self.random_crop = random_crop

        self.min_length_sec = min_length_sec
        self.max_length_sec = max_length_sec

        # tokenizers: dict mapping metadata key -> (tokenizer, max_length)
        # If provided, text fields will be pre-tokenized in DataLoader workers
        self.tokenizers = tokenizers

        print(f'Found {len(self.filenames)} files')

    def __len__(self):
        return len(self.filenames)

    def _get_silence_for_file(self, latent_filename):
        """Return the silence latent for the dataset that contains this file, or None."""
        for path, silence in self.silence_latents.items():
            if path in latent_filename:
                return silence
        return None

    def __getitem__(self, idx):
        latent_filename, md_filename = self.filenames[idx]
        try:
            latents = torch.from_numpy(np.load(latent_filename)) # [C, N]

            with open(md_filename, "r") as f:
                try:
                    info = json.load(f)
                except:
                    raise Exception(f"Couldn't load metadata file {md_filename}")

            info["latent_filename"] = latent_filename

            if self.latent_crop_length is not None:
                stored_length = latents.shape[1]

                if stored_length > self.latent_crop_length:
                    # Crop to latent_crop_length (existing logic)
                    # Get the last index from the padding mask, the index of the last 1 in the sequence
                    last_ix = len(info["padding_mask"]) - 1 - info["padding_mask"][::-1].index(1)

                    if self.random_crop and last_ix > self.latent_crop_length:
                        start = random.randint(0, last_ix - self.latent_crop_length)
                    else:
                        start = 0

                    latents = latents[:, start:start+self.latent_crop_length]
                    info["padding_mask"] = info["padding_mask"][start:start+self.latent_crop_length]
                    info["latent_crop_start"] = start

                elif stored_length < self.latent_crop_length:
                    # Pad with silence latent to reach latent_crop_length
                    pad_needed = self.latent_crop_length - stored_length
                    silence = self._get_silence_for_file(latent_filename)

                    if silence is not None:
                        # Slice or tile silence latent to cover pad_needed frames
                        if silence.shape[1] >= pad_needed:
                            silence_pad = silence[:, :pad_needed]
                        else:
                            silence_pad = np.tile(silence, (1, (pad_needed // silence.shape[1]) + 1))[:, :pad_needed]
                        latents = torch.cat([latents, torch.from_numpy(silence_pad)], dim=1)
                    else:
                        # No silence latent available — zero-pad as fallback
                        latents = torch.nn.functional.pad(latents, (0, pad_needed))

                    # Build padding_mask: valid frames from stored mask, zeros for padding
                    info["padding_mask"] = info["padding_mask"][:stored_length] + [0] * pad_needed
                    info["latent_crop_start"] = 0

                else:
                    # Exact match
                    info["latent_crop_start"] = 0

                info["latent_crop_length"] = self.latent_crop_length

            info["padding_mask"] = [torch.tensor(info["padding_mask"])]

            seconds_total = info.get("seconds_total")

            if seconds_total is not None and self.min_length_sec is not None and seconds_total < self.min_length_sec:
                return self[random.randrange(len(self))]

            if seconds_total is not None and self.max_length_sec is not None and seconds_total > self.max_length_sec:
                return self[random.randrange(len(self))]

            for custom_md_path in self.custom_metadata_fns.keys():
                if custom_md_path in latent_filename:
                    custom_metadata_fn = dill.loads(self.custom_metadata_fns[custom_md_path])
                    custom_metadata = custom_metadata_fn(info, latents)
                    info.update(custom_metadata)

                if "__reject__" in info and info["__reject__"]:
                    return self[random.randrange(len(self))]

                if "__replace__" in info and info["__replace__"] is not None:
                    # Replace the latents with the new latents if the custom metadata function returns a new set of latents
                    latents = info["__replace__"]

            info = maybe_build_ttm_condition(info)

            info["audio"] = latents

            # Pre-tokenize text fields in DataLoader workers to avoid
            # CPU contention with the main training thread
            if self.tokenizers is not None:
                for key, (tokenizer, max_length) in self.tokenizers.items():
                    if key in info and isinstance(info[key], str):
                        # Save raw text before replacing with tokens (needed by CLAP and other text-based losses)
                        info[f"{key}_text"] = info[key]
                        encoded = tokenizer(
                            info[key],
                            truncation=True,
                            max_length=max_length,
                            padding="max_length",
                            return_tensors="pt",
                        )
                        info[key] = {
                            "input_ids": encoded["input_ids"].squeeze(0),
                            "attention_mask": encoded["attention_mask"].squeeze(0),
                        }

            return (latents, info)
        except Exception as e:
            print(f'Couldn\'t load file {latent_filename}: {e}')
            return self[random.randrange(len(self))]

# get_dbmax and is_silence copied from https://github.com/drscotthawley/aeiou/blob/main/aeiou/core.py under Apache 2.0 License
# License can be found in LICENSES/LICENSE_AEIOU.txt
def get_dbmax(
    audio,       # torch tensor of (multichannel) audio
    ):
    "finds the loudest value in the entire clip and puts that into dB (full scale)"
    return 20*torch.log10(torch.flatten(audio.abs()).max()).cpu().numpy()

def is_silence(
    audio,       # torch tensor of (multichannel) audio
    thresh=-60,  # threshold in dB below which we declare to be silence
    ):
    "checks if entire clip is 'silence' below some dB threshold"
    dBmax = get_dbmax(audio)
    return dBmax < thresh


def audio_decoder(key, value):
    ext = key.split(".")[-1].lower()
    if ext in AUDIO_KEYS:
        return torchaudio.load(io.BytesIO(value))
    return None


def get_s3_contents(
    dataset_path,
    s3_url_prefix=None,
    filter="",
    recursive=True,
    debug=False,
    profile=None,
    s3_streaming_config=None,
):
    s3_streaming_config = normalize_s3_streaming_config(s3_streaming_config)

    if dataset_path != "" and not dataset_path.endswith("/"):
        dataset_path += "/"

    bucket_path = posixpath.join(s3_url_prefix or "", dataset_path)
    cmd = build_aws_cli_base_cmd(s3_streaming_config, profile=profile)
    cmd.extend(["s3", "ls", bucket_path])
    if recursive:
        cmd.append("--recursive")

    run_ls = subprocess.run(
        cmd,
        capture_output=True,
        check=True,
        timeout=s3_streaming_config["ls_timeout_sec"],
        env=build_aws_cli_env(s3_streaming_config),
    )
    contents = run_ls.stdout.decode("utf-8").split("\n")
    contents = [x.strip() for x in contents if x]
    contents = [
        re.sub(r"^\S+\s+\S+\s+\d+\s+", "", x)
        if re.match(r"^\S+\s+\S+\s+\d+\s+", x)
        else x
        for x in contents
    ]
    contents = [
        posixpath.join(s3_url_prefix or "", x) for x in contents if not x.endswith("/")
    ]

    if filter:
        contents = [x for x in contents if filter in x]

    if recursive:
        main_dir = "/".join(bucket_path.split("/")[3:])
        contents = [x.replace(f"{main_dir}", "").replace("//", "/") for x in contents]

    if debug:
        print("contents = \n", contents)

    return contents


def get_all_s3_urls(
    names=None,
    subsets=None,
    s3_url_prefix=None,
    recursive=True,
    filter_str="tar",
    debug=False,
    profiles=None,
    s3_streaming_configs=None,
):
    names = names or []
    subsets = subsets or [""]
    profiles = profiles or {}
    s3_streaming_configs = s3_streaming_configs or {}
    urls = []

    def make_s3_cp_source(name, subset, tar):
        tar = tar.strip()
        if tar.startswith("s3://"):
            return tar
        base = name if s3_url_prefix is None else posixpath.join(s3_url_prefix, name)
        if tar.startswith("/"):
            if subset:
                tar = tar.lstrip("/")
                if tar.startswith(f"{subset}/"):
                    tar = tar[len(subset) + 1:]
            else:
                return posixpath.join(base.rstrip("/"), tar.lstrip("/"))
        return posixpath.join(base, subset, tar)

    for name in names:
        contents_str = name if s3_url_prefix is None else posixpath.join(s3_url_prefix, name)
        for subset in subsets:
            subset_str = posixpath.join(contents_str, subset)
            tar_list = get_s3_contents(
                subset_str,
                s3_url_prefix=None,
                recursive=recursive,
                filter=filter_str,
                debug=debug,
                profile=profiles.get(name),
                s3_streaming_config=s3_streaming_configs.get(name),
            )
            for tar in tar_list:
                s3_path = make_s3_cp_source(name, subset, tar)
                urls.append(
                    build_s3_pipe_request(
                        s3_path,
                        profile=profiles.get(name),
                        s3_streaming_config=s3_streaming_configs.get(name),
                    )
                )
    return urls


def log_and_continue(exn):
    print(f"Handling webdataset error ({repr(exn)}). Ignoring.")
    return True


def is_valid_sample(sample):
    if sample is None or not isinstance(sample, dict):
        return False

    json_data = sample.get("json")
    audio_data = sample.get("audio")
    has_json = isinstance(json_data, dict)
    has_audio = audio_data is not None
    is_pre_encoded = sample.get("__pre_encoded__", False)
    is_silent = has_audio and (not is_pre_encoded) and is_silence(audio_data)
    is_rejected = has_json and json_data.get("__reject__", False)
    has_prompt = has_json and is_valid_prompt_value(json_data.get("prompt"))

    return has_json and has_audio and has_prompt and not is_silent and not is_rejected


class WebDatasetDataLoader:
    def __init__(
        self,
        datasets: List[LocalWebDatasetConfig],
        batch_size,
        sample_size,
        sample_rate=48000,
        num_workers=8,
        epoch_steps=1000,
        random_crop=True,
        force_channels="stereo",
        augment_phase=True,
        pre_encoded=False,
        latent_crop_length=None,
        latent_extension="npy",
        min_length_sec=None,
        max_length_sec=None,
        resampled_shards=True,
        **data_loader_kwargs,
    ):
        wds = _require_webdataset()

        self.datasets = datasets
        self.sample_size = sample_size
        self.sample_rate = sample_rate
        self.random_crop = random_crop
        self.force_channels = force_channels
        self.augment_phase = augment_phase
        self.pre_encoded = pre_encoded
        self.latent_crop_length = latent_crop_length
        self.latent_extension = latent_extension.lower().lstrip(".")
        self.min_length_sec = min_length_sec
        self.max_length_sec = max_length_sec

        urls = [dataset.load_data_urls() for dataset in datasets]
        urls = [url for dataset_urls in urls for url in dataset_urls]
        if not urls:
            raise ValueError("No WebDataset .tar shards found")
        random.shuffle(urls)

        self.dataset = wds.DataPipeline(
            wds.ResampledShards(urls) if resampled_shards else wds.SimpleShardList(urls),
            wds.tarfile_to_samples(handler=log_and_continue),
            wds.decode(self._decoder(), handler=log_and_continue),
            wds.map(self.wds_preprocess, handler=log_and_continue),
            wds.select(is_valid_sample),
            wds.to_tuple("audio", "json", handler=log_and_continue),
            wds.batched(batch_size, partial=False, collation_fn=collation_fn),
        )

        if resampled_shards:
            steps_per_worker = epoch_steps // num_workers if num_workers > 0 else epoch_steps
            self.dataset = self.dataset.with_epoch(max(1, steps_per_worker))

        def worker_init_fn(worker_id):
            torch.multiprocessing.set_sharing_strategy("file_system")

        self.data_loader = wds.WebLoader(
            self.dataset,
            batch_size=None,
            num_workers=num_workers,
            worker_init_fn=worker_init_fn,
            **data_loader_kwargs,
        )

    def __iter__(self):
        return iter(self.data_loader)

    def __len__(self):
        return len(self.data_loader)

    def _decoder(self):
        if not self.pre_encoded:
            return audio_decoder

        latent_extension = self.latent_extension

        def latent_decoder(key, value):
            ext = key.split(".")[-1].lower()
            if ext == latent_extension:
                return np.lib.format.read_array(io.BytesIO(value))
            return None

        return latent_decoder

    def _apply_pre_encoded_rules(self, latents, info, source_path, custom_metadata_fn):
        info = dict(info)
        info["latent_filename"] = source_path

        if self.latent_crop_length is not None:
            stored_length = latents.shape[1]
            padding_mask = info.get("padding_mask", [1] * stored_length)

            if stored_length > self.latent_crop_length:
                last_ix = len(padding_mask) - 1 - padding_mask[::-1].index(1)
                if self.random_crop and last_ix > self.latent_crop_length:
                    start = random.randint(0, last_ix - self.latent_crop_length)
                else:
                    start = 0
                latents = latents[:, start : start + self.latent_crop_length]
                info["padding_mask"] = padding_mask[start : start + self.latent_crop_length]
                info["latent_crop_start"] = start
            elif stored_length < self.latent_crop_length:
                pad_needed = self.latent_crop_length - stored_length
                latents = torch.nn.functional.pad(latents, (0, pad_needed))
                info["padding_mask"] = padding_mask[:stored_length] + [0] * pad_needed
                info["latent_crop_start"] = 0
            else:
                info["padding_mask"] = padding_mask
                info["latent_crop_start"] = 0

            info["latent_crop_length"] = self.latent_crop_length

        info["padding_mask"] = [torch.tensor(info.get("padding_mask", []))]

        seconds_total = info.get("seconds_total")
        if seconds_total is not None:
            if self.min_length_sec is not None and seconds_total < self.min_length_sec:
                info["__reject__"] = True
            if self.max_length_sec is not None and seconds_total > self.max_length_sec:
                info["__reject__"] = True

        if custom_metadata_fn is not None:
            custom_metadata = custom_metadata_fn(info, latents)
            info.update(custom_metadata)
            if "__replace__" in info and info["__replace__"] is not None:
                latents = info["__replace__"]

        info["audio"] = latents
        return latents, info

    def wds_preprocess(self, sample):
        metadata = sample.get("json")
        if not isinstance(metadata, dict):
            return None

        if "text" in metadata and "prompt" not in metadata:
            metadata["prompt"] = metadata["text"]

        if self.pre_encoded:
            found_key = ""
            for key in list(sample.keys()):
                if key.endswith(self.latent_extension):
                    found_key = key
                    break
            if not found_key:
                return None
            audio = torch.from_numpy(sample[found_key])
            sample["__pre_encoded__"] = True
        else:
            found_key = ""
            for key in sample.keys():
                for audio_key in AUDIO_KEYS:
                    if key.endswith(audio_key):
                        found_key = key
                        break
                if found_key:
                    break
            if not found_key:
                return None

            audio, in_sr = sample[found_key]
            if in_sr != self.sample_rate:
                resample_tf = T.Resample(in_sr, self.sample_rate)
                audio = resample_tf(audio)

            if self.sample_size is not None:
                pad_crop = PadCrop_Normalized_T(
                    self.sample_size,
                    self.sample_rate,
                    randomize=self.random_crop,
                )
                audio, t_start, t_end, seconds_start, seconds_total, padding_mask = pad_crop(audio)
                metadata["seconds_start"] = seconds_start
                metadata["seconds_total"] = seconds_total
                metadata["padding_mask"] = padding_mask
                metadata["timestamps"] = (t_start, t_end)

            if audio.shape[-1] == 0:
                audio = torch.zeros(1, 1)

            augs = torch.nn.Sequential(
                Stereo() if self.force_channels == "stereo" else torch.nn.Identity(),
                Mono() if self.force_channels == "mono" else torch.nn.Identity(),
                PhaseFlipper() if self.augment_phase else torch.nn.Identity(),
            )
            audio = augs(audio).clamp(-1, 1)

        matched_dataset = None
        for dataset in self.datasets:
            if dataset.path not in sample.get("__url__", ""):
                continue
            matched_dataset = dataset
            if self.pre_encoded:
                audio, metadata = self._apply_pre_encoded_rules(
                    audio,
                    metadata,
                    sample.get("__key__", ""),
                    dataset.custom_metadata_fn,
                )
            elif dataset.custom_metadata_fn is not None:
                metadata.update(dataset.custom_metadata_fn(metadata, audio))
            break

        if self.pre_encoded and matched_dataset is None:
            audio, metadata = self._apply_pre_encoded_rules(
                audio,
                metadata,
                sample.get("__key__", ""),
                None,
            )

        metadata = maybe_build_ttm_condition(metadata)

        metadata["audio"] = audio
        sample["audio"] = audio
        sample["json"] = metadata

        return sample


def create_dataloader_from_config(
    dataset_config,
    batch_size,
    sample_size,
    sample_rate,
    num_workers=4,
    audio_channels=2,
    return_valid=False,
):
    dataset_type = dataset_config.get("dataset_type")
    if dataset_type is None:
        raise ValueError("Dataset config must include dataset_type")

    force_channels = "mono" if audio_channels == 1 else "stereo"

    def _with_valid(train_loader, build_loader):
        if not return_valid:
            return train_loader

        valid_entries = dataset_config.get("datasets_valid", [])
        if not valid_entries:
            return train_loader, []

        valid_config = dict(dataset_config)
        valid_config["datasets"] = valid_entries
        valid_config["random_crop"] = dataset_config.get("random_crop_valid", False)
        valid_config["shuffle"] = dataset_config.get("shuffle_valid", False)
        valid_config["drop_last"] = dataset_config.get("drop_last_valid", True)
        if "epoch_steps_valid" in dataset_config:
            valid_config["epoch_steps"] = dataset_config["epoch_steps_valid"]
        if "resampled_shards_valid" in dataset_config:
            valid_config["resampled_shards"] = dataset_config["resampled_shards_valid"]

        return train_loader, [build_loader(valid_config)]

    if dataset_type == "audio_dir":
        def build_audio_dir_loader(config):
            configs = [
                LocalDatasetConfig(
                    id=entry["id"],
                    path=entry["path"],
                    keywords=entry.get("keywords"),
                    filelist_path=entry.get("filelist_path"),
                    weight=entry.get("weight", 1.0),
                )
                for entry in config.get("datasets", [])
            ]
            dataset = SampleDataset(
                configs,
                sample_size=sample_size,
                sample_rate=sample_rate,
                random_crop=config.get("random_crop", True),
                force_channels=force_channels,
                volume_norm=config.get("volume_norm", False),
                volume_norm_param=tuple(config.get("volume_norm_param", (-16, 2))),
                strip_silence=config.get("strip_silence", False),
                pad=config.get("pad", True),
            )
            return torch.utils.data.DataLoader(
                dataset,
                batch_size=batch_size,
                shuffle=config.get("shuffle", True),
                num_workers=num_workers,
                drop_last=config.get("drop_last", True),
                collate_fn=collation_fn,
            )

        return _with_valid(build_audio_dir_loader(dataset_config), build_audio_dir_loader)

    if dataset_type == "pre_encoded":
        def build_pre_encoded_loader(config):
            configs = [
                LatentDatasetConfig(
                    id=entry["id"],
                    path=entry["path"],
                    latent_extension=entry.get("latent_extension", config.get("latent_extension", "npy")),
                    filelist_path=entry.get("filelist_path"),
                    weight=entry.get("weight", 1.0),
                )
                for entry in config.get("datasets", [])
            ]
            dataset = PreEncodedDataset(
                configs,
                latent_crop_length=config.get("latent_crop_length"),
                min_length_sec=config.get("min_length_sec"),
                max_length_sec=config.get("max_length_sec"),
                random_crop=config.get("random_crop", False),
            )
            return torch.utils.data.DataLoader(
                dataset,
                batch_size=batch_size,
                shuffle=config.get("shuffle", True),
                num_workers=num_workers,
                drop_last=config.get("drop_last", True),
                collate_fn=collation_fn,
            )

        return _with_valid(build_pre_encoded_loader(dataset_config), build_pre_encoded_loader)

    if dataset_type in ("s3", "wds"):
        def build_webdataset_loader(config):
            configs = []
            s3_streaming_defaults = config.get("s3_streaming", {})
            for entry in config.get("datasets", []):
                if dataset_type == "s3":
                    s3_streaming_config = dict(s3_streaming_defaults)
                    s3_streaming_config.update(entry.get("s3_streaming", {}))
                    configs.append(
                        S3DatasetConfig(
                            id=entry["id"],
                            s3_path=entry["s3_path"],
                            profile=entry.get("profile"),
                            s3_streaming_config=s3_streaming_config,
                        )
                    )
                else:
                    configs.append(LocalWebDatasetConfig(id=entry["id"], path=entry["path"]))

            pre_encoded = config.get("pre_encoded", False)
            return WebDatasetDataLoader(
                configs,
                batch_size=batch_size,
                sample_size=sample_size,
                sample_rate=sample_rate,
                num_workers=num_workers,
                epoch_steps=config.get("epoch_steps", 2000),
                random_crop=config.get("random_crop", True),
                force_channels=force_channels,
                pre_encoded=pre_encoded,
                latent_crop_length=config.get("latent_crop_length"),
                latent_extension=config.get("latent_extension", "npy"),
                min_length_sec=config.get("min_length_sec"),
                max_length_sec=config.get("max_length_sec"),
                resampled_shards=config.get("resampled_shards", True),
            )

        return _with_valid(build_webdataset_loader(dataset_config), build_webdataset_loader)

    raise ValueError(f"Unknown dataset_type: {dataset_type}")


def collation_fn(samples):
        batched = list(zip(*samples))
        result = []
        for b in batched:
            if isinstance(b[0], (int, float)):
                b = np.array(b)
            elif isinstance(b[0], torch.Tensor):
                b = torch.stack(b)
            elif isinstance(b[0], np.ndarray):
                b = np.array(b)
            else:
                b = b
            result.append(b)
        return result
