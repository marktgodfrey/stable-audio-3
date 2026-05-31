import argparse
import copy
import json
from pathlib import Path

from stable_audio_3.model_configs import models


def load_config(source_config, source_model):
    if source_config is not None:
        with open(source_config) as f:
            return json.load(f)

    config_path, _ = models[source_model].resolve()
    with open(config_path) as f:
        return json.load(f)


def replace_if_equal(config, key, old_value, new_value):
    if config.get(key) == old_value:
        config[key] = new_value


def make_large_config(config, old_dim=1536, new_dim=2048, depth=26, num_heads=32):
    config = copy.deepcopy(config)

    diffusion = config["model"]["diffusion"]["config"]
    diffusion["embed_dim"] = new_dim
    diffusion["depth"] = depth
    diffusion["num_heads"] = num_heads

    # Keep dimensions internally consistent when the medium config ties them to d.
    for key in (
        "cond_token_dim",
        "global_cond_dim",
        "input_concat_dim",
        "prepend_cond_dim",
        "timestep_embed_dim",
    ):
        replace_if_equal(diffusion, key, old_dim, new_dim)

    conditioning = config["model"].get("conditioning")
    if conditioning is not None:
        replace_if_equal(conditioning, "cond_dim", old_dim, new_dim)
        for conditioner in conditioning.get("configs", []):
            conditioner_config = conditioner.get("config", {})
            replace_if_equal(conditioner_config, "output_dim", old_dim, new_dim)

    config.setdefault("metadata", {})
    config["metadata"]["derived_from"] = "stable-audio-3-medium"
    config["metadata"]["large_like_config"] = {
        "embed_dim": new_dim,
        "depth": depth,
        "num_heads": num_heads,
        "head_dim": new_dim // num_heads,
        "note": "Architecture derived from Table 2; weights are not converted.",
    }

    return config


def main():
    parser = argparse.ArgumentParser(
        description="Create a large-like Stable Audio 3 config from a medium config."
    )
    parser.add_argument("--source_config", default=None)
    parser.add_argument("--source_model", choices=["medium", "medium-base"], default="medium-base")
    parser.add_argument("--output", required=True)
    parser.add_argument("--old_dim", type=int, default=1536)
    parser.add_argument("--embed_dim", type=int, default=2048)
    parser.add_argument("--depth", type=int, default=26)
    parser.add_argument("--num_heads", type=int, default=32)
    args = parser.parse_args()

    config = load_config(args.source_config, args.source_model)
    config = make_large_config(
        config,
        old_dim=args.old_dim,
        new_dim=args.embed_dim,
        depth=args.depth,
        num_heads=args.num_heads,
    )

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w") as f:
        json.dump(config, f, indent=2)
        f.write("\n")

    print(f"Wrote {output}")


if __name__ == "__main__":
    main()
