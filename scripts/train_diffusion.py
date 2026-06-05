"""
Train or continue training a full Stable Audio 3 diffusion model.

Dataset config example for S3 WebDataset shards:
{
  "dataset_type": "s3",
  "pre_encoded": true,
  "datasets": [
    {"id": "train", "s3_path": "s3://my-bucket/datasets/latents/train/"}
  ],
  "epoch_steps": 2000,
  "random_crop": true,
  "s3_streaming": {
    "stream_idle_timeout_sec": 300
  }
}
"""

import argparse
import copy
import json
import os
from datetime import timedelta

import pytorch_lightning as pl
import torch
from safetensors.torch import load_file

from stable_audio_3.data.dataset import create_dataloader_from_config
from stable_audio_3.factory import create_diffusion_cond_from_config
from stable_audio_3.loading_utils import copy_state_dict, load_autoencoder
from stable_audio_3.model_configs import ae_models, base_models
from stable_audio_3.training.diffusion import DiffusionCondInpaintDemoCallback, DiffusionCondTrainingWrapper


class ExceptionCallback(pl.Callback):
    def on_exception(self, trainer, module, err):
        print(f"{type(err).__name__}: {err}")


def load_model(model_name, model_config_path, checkpoint_path, init_from_pretrained=True):
    if model_name is not None:
        if model_name not in base_models:
            raise ValueError(f"Unknown model '{model_name}', valid: {list(base_models)}")
        resolved_config, resolved_checkpoint = base_models[model_name].resolve()
        model_config_path = model_config_path or resolved_config
        if init_from_pretrained:
            checkpoint_path = checkpoint_path or resolved_checkpoint

    if model_config_path is None:
        raise ValueError("Provide --model or --model_config")

    with open(model_config_path) as f:
        model_config = json.load(f)

    model = create_diffusion_cond_from_config(model_config)
    if checkpoint_path is not None:
        copy_state_dict(model, load_file(checkpoint_path))

    model.train()
    return model, model_config


def load_pretrained_pretransform(model, pretransform_model_name):
    if pretransform_model_name is None:
        return

    if model.pretransform is None:
        raise ValueError(
            f"Cannot load pretransform model '{pretransform_model_name}': model has no pretransform"
        )

    if pretransform_model_name not in ae_models:
        raise ValueError(
            f"Unknown pretransform model '{pretransform_model_name}', valid: {list(ae_models)}"
        )

    cfg = ae_models[pretransform_model_name]
    local_config, local_ckpt = cfg.resolve()
    print(
        f"Loading pretrained pretransform '{pretransform_model_name}' "
        f"from {local_ckpt}"
    )
    autoencoder = load_autoencoder(local_config, local_ckpt, device="cpu")
    copy_state_dict(model.pretransform.model, autoencoder.state_dict())
    model.pretransform.enable_grad = False
    model.pretransform.eval().requires_grad_(False)


def train(args):
    torch._dynamo.config.capture_scalar_outputs = True
    torch.set_float32_matmul_precision("high")
    pl.seed_everything(args.seed, workers=True)

    model, model_config = load_model(
        args.model,
        args.model_config,
        args.checkpoint,
        args.init_from_pretrained,
    )
    training_config = model_config.get("training", {})
    pretransform_model = args.pretransform_model or training_config.get("pretransform_model")
    load_pretrained_pretransform(model, pretransform_model)
    use_ema = args.use_ema if args.use_ema is not None else training_config.get("use_ema", False)
    timestep_sampler = (
        args.timestep_sampler
        if args.timestep_sampler is not None
        else training_config.get("timestep_sampler", "trunc_logit_normal")
    )
    mask_loss_weight = (
        args.mask_loss_weight
        if args.mask_loss_weight is not None
        else training_config.get("mask_loss_weight", 1.0)
    )
    silence_extension_scale_seconds = (
        args.silence_extension_scale_seconds
        if args.silence_extension_scale_seconds is not None
        else training_config.get("silence_extension_scale_seconds", 4.0)
    )
    ot_coupling = args.ot_coupling if args.ot_coupling is not None else training_config.get("ot_coupling", True)
    inpainting_enabled = args.inpainting if args.inpainting is not None else bool(training_config.get("inpainting"))
    log_loss_info = (
        args.log_loss_info
        if args.log_loss_info is not None
        else training_config.get("log_loss_info", False)
    )

    sample_rate = model_config.get("sample_rate", getattr(model, "sample_rate", 44100))
    ds_ratio = model.pretransform.downsampling_ratio if model.pretransform is not None else 1
    sample_size = (int(args.duration * sample_rate) // ds_ratio) * ds_ratio

    with open(args.dataset_config) as f:
        dataset_config = json.load(f)

    if dataset_config.get("pre_encoded", False) and dataset_config.get("latent_crop_length") is None:
        dataset_config["latent_crop_length"] = sample_size // ds_ratio

    dataloader, valid_dataloaders = create_dataloader_from_config(
        dataset_config,
        batch_size=args.batch_size,
        sample_size=sample_size,
        sample_rate=sample_rate,
        num_workers=args.num_workers,
        audio_channels=args.audio_channels,
        return_valid=True,
    )

    optimizer_config = None if args.ignore_model_optimizer_config else training_config.get("optimizer_configs")
    if optimizer_config is not None:
        optimizer_config = copy.deepcopy(optimizer_config)
    else:
        optimizer_config = {
            "diffusion": {
                "optimizer": {
                    "type": "AdamW",
                    "config": {
                        "lr": args.lr,
                        "weight_decay": args.weight_decay,
                        "betas": [0.9, 0.95],
                    },
                }
            }
        }

    training_wrapper = DiffusionCondTrainingWrapper(
        model,
        mask_loss_weight=mask_loss_weight,
        mask_padding_attention=args.mask_padding_attention,
        silence_extension_scale_seconds=silence_extension_scale_seconds,
        use_ema=use_ema,
        log_loss_info=log_loss_info,
        optimizer_configs=optimizer_config,
        pre_encoded=dataset_config.get("pre_encoded", False),
        timestep_sampler=timestep_sampler,
        timestep_sampler_options={},
        inpainting_config={"mask_kwargs": {"mask_type_probabilities": [0.1, 0.8, 0.1]}}
        if inpainting_enabled
        else None,
        use_effective_length_for_schedule=args.use_effective_length_for_schedule,
        sample_rate=sample_rate,
        sample_size=sample_size,
        log_every_n_steps=args.log_every,
        ot_coupling=ot_coupling,
    )

    logger = None
    if args.logger == "wandb":
        logger = pl.loggers.WandbLogger(project=args.name)
        logger.watch(training_wrapper)
    elif args.logger == "comet":
        logger = pl.loggers.CometLogger(project=args.name)
    elif args.logger == "csv":
        logger = pl.loggers.CSVLogger(args.save_dir, name=args.name)

    checkpoint_dir = os.path.join(args.save_dir, args.name, "checkpoints")
    ckpt_callback = pl.callbacks.ModelCheckpoint(
        every_n_train_steps=args.checkpoint_every,
        dirpath=checkpoint_dir,
        save_top_k=args.save_top_k,
        monitor="global_step",
        mode="max",
    )

    callbacks = [
        ckpt_callback,
        ExceptionCallback(),
        pl.callbacks.ModelSummary(max_depth=2),
    ]

    if args.checkpoint_time_interval_minutes > 0:
        timed_ckpt_callback = pl.callbacks.ModelCheckpoint(
            train_time_interval=timedelta(minutes=args.checkpoint_time_interval_minutes),
            dirpath=checkpoint_dir,
            save_last="link",
        )
        callbacks.append(timed_ckpt_callback)

    demo_config = training_config.get("demo", {})
    demo_every = args.demo_every if args.demo_every is not None else demo_config.get("demo_every", 500)
    validation_every = (
        args.validation_every
        if args.validation_every is not None
        else demo_every if demo_every and demo_every > 0 else args.checkpoint_every
    )

    if demo_every and demo_every > 0:
        configured_num_demos = args.num_demos if args.num_demos is not None else demo_config.get("num_demos", 4)
        demo_source_loader = valid_dataloaders[0] if valid_dataloaders else dataloader

        callbacks.append(
            DiffusionCondInpaintDemoCallback(
                demo_every=demo_every,
                sample_size=sample_size,
                sample_rate=sample_rate,
                demo_steps=args.demo_steps if args.demo_steps is not None else demo_config.get("demo_steps", 50),
                num_demos=configured_num_demos,
                demo_cfg_scales=args.demo_cfg_scales or demo_config.get("demo_cfg_scales", [2, 4, 7]),
                demo_conditioning=demo_config.get("demo_cond", []),
                inpaint_demo_config=demo_config.get("inpaint_demo_config"),
                demo_dl=demo_source_loader,
            )
        )

    trainer_kwargs = {}
    run_validation = bool(valid_dataloaders) and validation_every and validation_every > 0
    if run_validation:
        trainer_kwargs.update(
            check_val_every_n_epoch=None,
            val_check_interval=validation_every,
        )

    trainer = pl.Trainer(
        devices="auto",
        accelerator="auto",
        strategy=args.strategy,
        precision=args.precision,
        accumulate_grad_batches=args.accumulate_grad_batches,
        callbacks=callbacks,
        logger=logger,
        log_every_n_steps=1,
        max_steps=args.steps,
        default_root_dir=args.save_dir,
        gradient_clip_val=args.gradient_clip_val or None,
        num_sanity_val_steps=0,
        **trainer_kwargs,
    )

    trainer.fit(
        training_wrapper,
        train_dataloaders=dataloader,
        val_dataloaders=valid_dataloaders if run_validation else None,
        ckpt_path=args.resume_from_checkpoint,
    )

    if args.export_path:
        os.makedirs(os.path.dirname(args.export_path) or ".", exist_ok=True)
        training_wrapper.export_model(args.export_path, use_safetensors=args.export_path.endswith(".safetensors"))


def main():
    p = argparse.ArgumentParser(description="Train a full Stable Audio 3 diffusion model")
    p.add_argument("--model", choices=list(base_models), default=None)
    p.add_argument("--model_config", default=None)
    p.add_argument("--checkpoint", default=None)
    p.add_argument(
        "--pretransform_model",
        choices=list(ae_models),
        default=None,
        help=(
            "Initialize the model pretransform from a public SAME autoencoder "
            "checkpoint, e.g. same-l. Can also be set as training.pretransform_model."
        ),
    )
    p.add_argument(
        "--init_from_pretrained",
        "--init-from-pretrained",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    p.add_argument("--resume_from_checkpoint", default=None)
    p.add_argument("--dataset_config", required=True)
    p.add_argument("--duration", type=float, default=380.0)
    p.add_argument("--audio_channels", type=int, choices=[1, 2], default=2)
    p.add_argument("--lr", type=float, default=1e-5)
    p.add_argument("--weight_decay", type=float, default=0.01)
    p.add_argument("--ignore_model_optimizer_config", action="store_true")
    p.add_argument("--steps", type=int, default=10_000)
    p.add_argument("--batch_size", type=int, default=1)
    p.add_argument("--num_workers", type=int, default=8)
    p.add_argument("--accumulate_grad_batches", type=int, default=1)
    p.add_argument("--gradient_clip_val", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--logger", choices=["wandb", "comet", "csv", "none"], default="csv")
    p.add_argument("--name", default="diffusion-train")
    p.add_argument("--save_dir", default="./training_runs")
    p.add_argument("--checkpoint_every", type=int, default=500)
    p.add_argument("--save_top_k", type=int, default=10)
    p.add_argument("--checkpoint_time_interval_minutes", type=int, default=60)
    p.add_argument("--log_every", type=int, default=100)
    p.add_argument("--demo_every", type=int, default=None)
    p.add_argument("--demo_steps", type=int, default=None)
    p.add_argument("--num_demos", type=int, default=None)
    p.add_argument("--demo_cfg_scales", type=float, nargs="+", default=None)
    p.add_argument("--validation_every", type=int, default=None)
    p.add_argument("--export_path", default=None)
    p.add_argument("--precision", default="bf16-mixed")
    p.add_argument("--strategy", default="auto")
    p.add_argument(
        "--timestep_sampler",
        choices=["uniform", "logit_normal", "trunc_logit_normal", "log_snr", "log_snr_uniform"],
        default=None,
    )
    p.add_argument("--mask_loss_weight", type=float, default=None)
    p.add_argument("--mask_padding_attention", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--silence_extension_scale_seconds", type=float, default=None)
    p.add_argument("--use_effective_length_for_schedule", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--inpainting", action=argparse.BooleanOptionalAction, default=None)
    p.add_argument("--use_ema", action=argparse.BooleanOptionalAction, default=None)
    p.add_argument("--log_loss_info", action=argparse.BooleanOptionalAction, default=None)
    p.add_argument("--ot_coupling", action=argparse.BooleanOptionalAction, default=None)
    args = p.parse_args()

    train(args)


if __name__ == "__main__":
    main()
