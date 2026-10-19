#!/usr/bin/env python
# coding=utf-8
# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import sys
import argparse
import logging
import math
import os
import random
from datetime import datetime
from pathlib import Path
from omegaconf import OmegaConf
import torch
import transformers
from accelerate import Accelerator, DistributedType
from accelerate.logging import get_logger
from accelerate.utils import (
    DistributedDataParallelKwargs,
    ProjectConfiguration,
    set_seed,
)
from tqdm.auto import tqdm
from torchvision.utils import save_image
import diffusers
from transformers import AutoTokenizer, CLIPTextModel
from diffusers import (
    AutoencoderKL, DDPMScheduler, UNet2DConditionModel
)
from diffusers.optimization import get_scheduler
from diffusers.training_utils import (
    free_memory,
)
from diffusers.utils.torch_utils import is_compiled_module
import torch.nn.functional as F
import warnings
SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))
from dataset.my_dataset import PairedDataset
from diffusers.utils.import_utils import is_xformers_available
from peft import LoraConfig, PeftModel
from adabridge.trajectory_control import TrajectoryControlPredictor
import copy

warnings.filterwarnings("ignore")

logger = get_logger(__name__)

def encode_images(pixels: torch.Tensor, vae: torch.nn.Module):
    pixel_latents = vae.encode(pixels.to(vae.dtype)).latent_dist.sample()
    pixel_latents = pixel_latents* vae.config.scaling_factor
    return pixel_latents

def set_vae_encoder_lora(vae_encoder, rank):
    target_modules = [
        "conv1",
        "conv2",
        "conv_in",
        "conv_shortcut",
        "conv",
        "conv_out",
        "to_k",
        "to_q",
        "to_v",
        "to_out.0",
    ]

    vae_encoder_lora_config = LoraConfig(
        r=rank,
        lora_alpha=rank,
        target_modules=target_modules,
        init_lora_weights="gaussian",
    )
    vae_encoder = PeftModel(
        model=vae_encoder,
        peft_config=vae_encoder_lora_config,
        adapter_name="vae_encoder_lora_adapter",
    )
    vae_encoder.print_trainable_parameters()
    return vae_encoder

def set_unet_lora(unet, rank):
    target_modules = [
        "conv1",
        "conv2",
        "conv_in",
        "conv_shortcut",
        "conv",
        "conv_out",
        "to_k",
        "to_q",
        "to_v",
        "to_out.0",
    ]

    # now we will add new LoRA weights to the attention layers
    unet_lora_config = LoraConfig(
        r=rank,
        lora_alpha=rank,
        target_modules=target_modules,
        init_lora_weights="gaussian",
    )
    unet = PeftModel(
        model=unet,
        peft_config=unet_lora_config,
        adapter_name="unet_lora_adapter",
    )
    unet.print_trainable_parameters()
    return unet

def parse_args():
    parser = argparse.ArgumentParser(description="AdaBridge-SR training")
    parser.add_argument(
        "--config",
        type=str,
        default="./configs/train.yml",
        help="path to config",
    )
    args = parser.parse_args()

    return args.config


def main():
    args = OmegaConf.load(parse_args())
    # Add a timestamp prefix to the experiment folder name on every run.
    output_dir_raw = str(args.output_dir)
    output_dir_parent = os.path.dirname(os.path.normpath(output_dir_raw))
    output_dir_name = os.path.basename(os.path.normpath(output_dir_raw))
    output_dir_timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    args.output_dir = os.path.join(output_dir_parent, f"{output_dir_timestamp}_{output_dir_name}")
    print(f"Resolved output_dir: {args.output_dir}")
    logging_dir = Path(args.output_dir, args.logging_dir)
    enable_trajectory_control = bool(args.get("enable_trajectory_control", False))
    enable_power_modulation = bool(args.get("enable_power_modulation", False))
    enable_sigma_map_modulation = bool(args.get("enable_sigma_map_modulation", False))
    enable_sigma_map_supervision = bool(args.get("enable_sigma_map_supervision", False))
    enable_sigma_eff_supervision = bool(args.get("enable_sigma_eff_supervision", False))
    power_min = float(args.get("power_min", 0.5))
    power_max = float(args.get("power_max", 1.5))
    lambda_sigma_map_supervision = float(args.get("lambda_sigma_map_supervision", 0.0))
    lambda_sigma_eff_supervision = float(args.get("lambda_sigma_eff_supervision", 0.0))
    power_ema_beta = float(args.get("power_ema_beta", 0.99))
    power_ema_warmup_steps = int(args.get("power_ema_warmup_steps", 500))

    if power_min > power_max:
        raise ValueError("`power_min` must be <= `power_max`.")
    if not (0.0 <= power_ema_beta < 1.0):
        raise ValueError("`power_ema_beta` must be in [0, 1).")
    if power_ema_warmup_steps < 0:
        raise ValueError("`power_ema_warmup_steps` must be >= 0.")
    if not enable_trajectory_control:
        if (
            enable_power_modulation
            or enable_sigma_map_modulation
            or enable_sigma_map_supervision
            or enable_sigma_eff_supervision
        ):
            warnings.warn(
                "enable_trajectory_control=False, forcing power/sigma_map modulation and supervision flags to False."
            )
        enable_power_modulation = False
        enable_sigma_map_modulation = False
        enable_sigma_map_supervision = False
        enable_sigma_eff_supervision = False
    if enable_sigma_eff_supervision and not enable_power_modulation:
        raise ValueError("`enable_sigma_eff_supervision` requires `enable_power_modulation=True`.")

    def resolve_init_weight_path(init_from_checkpoint, output_dir):
        if not init_from_checkpoint:
            return None

        if init_from_checkpoint == "latest":
            if not os.path.isdir(output_dir):
                return None
            dirs = [d for d in os.listdir(output_dir) if d.startswith("weight-")]
            dirs = sorted(
                dirs,
                key=lambda x: int(x.split("-")[1]) if x.split("-")[1].isdigit() else -1,
            )
            if len(dirs) == 0:
                return None
            init_path = os.path.join(output_dir, dirs[-1])
        else:
            init_path = init_from_checkpoint
            if not os.path.isabs(init_path):
                candidate = os.path.join(output_dir, init_path)
                if os.path.isdir(candidate):
                    init_path = candidate

        if not os.path.isdir(init_path):
            return None
        return init_path

    init_from_checkpoint = args.get("init_from_checkpoint", None)
    init_weight_path = resolve_init_weight_path(init_from_checkpoint, args.output_dir)
    if init_from_checkpoint and init_weight_path is None:
        raise FileNotFoundError(f"Initialization checkpoint not found: {init_from_checkpoint}")

    accelerator_project_config = ProjectConfiguration(
        project_dir=args.output_dir, logging_dir=logging_dir
    )
    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with="tensorboard",
        project_config=accelerator_project_config,
        kwargs_handlers=[ddp_kwargs],
    )
    logger.info(f"Using mixed_precision: {args.mixed_precision}")

    # Disable AMP for MPS.
    if torch.backends.mps.is_available():
        accelerator.native_amp = False

    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()

    # Always use a fixed seed for reproducible training.
    seed = args.get("seed", 123)
    seed = 123 if seed is None else int(seed)
    args.seed = seed
    set_seed(seed, device_specific=True)
    logger.info(f"Using fixed seed: {seed}")

    # Handle the repository creation
    if accelerator.is_main_process:
        if args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)
            OmegaConf.save(args, os.path.join(args.output_dir, "cfg.yml"))
            os.makedirs(logging_dir, exist_ok=True)
    accelerator.wait_for_everyone()
    accelerator.init_trackers(
        project_name=args.model,
    )
    if accelerator.is_main_process and len(accelerator.trackers) == 0:
        logger.warning(
            "No trackers initialized. Install tensorboard with `pip install tensorboard` to enable logging."
        )

    if not args.fixed_prompt_path:
        logger.info(f"Current prompt: {args.fixed_prompt}")
        tokenizer = AutoTokenizer.from_pretrained(args.sd_path, subfolder="tokenizer")
        text_encoder = CLIPTextModel.from_pretrained(args.sd_path, subfolder="text_encoder").to(accelerator.device)
        text_encoder.requires_grad_(False)
        text_encoder.eval()

        def encode_prompt(prompt_batch):
            """Encode text prompts into embeddings."""
            with torch.no_grad():
                prompt_embeds = [
                    text_encoder(
                        tokenizer(
                            caption,
                            max_length=tokenizer.model_max_length,
                            padding="max_length",
                            truncation=True,
                            return_tensors="pt",
                        ).input_ids.to(text_encoder.device)
                    )[0]
                    for caption in prompt_batch
                ]
            return torch.concat(prompt_embeds, dim=0)
        
        prompt_embeds = encode_prompt([args.fixed_prompt] * args.train_batch_size)
        del tokenizer
        del text_encoder
        free_memory()
    else:
        prompt_embeds = torch.load(args.fixed_prompt_path, map_location=accelerator.device)

    # timestep settings
    timestep_sequence = [int(t) for t in args.timestep_sequence]
    if len(timestep_sequence) == 0:
        raise ValueError("`timestep_sequence` must contain at least one timestep.")
    max_timestep = max(timestep_sequence + [0])
    if max_timestep <= 0:
        raise ValueError("`max(timestep_sequence)` must be > 0 for linear sigma mapping.")
    timestep_sigma_seq = {int(t): float(t) / float(max_timestep) for t in set(timestep_sequence)}
    noise_scheduler = DDPMScheduler.from_pretrained(args.sd_path, subfolder="scheduler")
    logger.info(f"Current {args.model} timestep_sequence = {timestep_sequence}")
    logger.info(f"Current {args.model} max_timestep = {max_timestep}")
    logger.info(f"Current {args.model} bridge_sigma = {args.bridge_sigma}")
    logger.info(f"Current {args.model} timestep_sigma_seq = {timestep_sigma_seq}")
    logger.info(
        f"Trajectory control flags | enable_trajectory_control={enable_trajectory_control}, "
        f"enable_power_modulation={enable_power_modulation}, "
        f"enable_sigma_map_modulation={enable_sigma_map_modulation}, "
        f"enable_sigma_map_supervision={enable_sigma_map_supervision}, "
        f"enable_sigma_eff_supervision={enable_sigma_eff_supervision}"
    )
    # fixed vae
    fixed_vae = AutoencoderKL.from_pretrained(args.sd_path, subfolder="vae")
    fixed_vae.requires_grad_(False)
    fixed_vae.eval()

    # lora_vae
    lora_vae = copy.deepcopy(fixed_vae)
    lora_vae.requires_grad_(False)
    del lora_vae.decoder
    free_memory()
    if init_weight_path is not None:
        lora_vae.encoder = PeftModel.from_pretrained(
            model=lora_vae.encoder,
            model_id=os.path.join(init_weight_path, "vae_encoder_lora_adapter"),
            adapter_name="vae_encoder_lora_adapter",
            is_trainable=True,
        )
        logger.info(f"Initialized lora_vae from {init_weight_path}")
        lora_vae.encoder.print_trainable_parameters()
    else:
        lora_vae.encoder = set_vae_encoder_lora(lora_vae.encoder, args.vae_lora_rank)
    lora_vae.train()

    # unet
    unet = UNet2DConditionModel.from_pretrained(args.sd_path, subfolder="unet")
    unet.requires_grad_(False)
    if init_weight_path is not None:
        unet = PeftModel.from_pretrained(
            model=unet,
            model_id=os.path.join(init_weight_path, "unet_lora_adapter"),
            adapter_name="unet_lora_adapter",
            is_trainable=True,
        )
        logger.info(f"Initialized unet from {init_weight_path}")
        unet.print_trainable_parameters()
    else:
        unet = set_unet_lora(unet, args.unet_lora_rank)
    unet.train()

    control_predictor = None
    if enable_trajectory_control:
        control_predictor = TrajectoryControlPredictor(
            in_channels=4,
            base_channels=int(args.get("control_base_channels", 64)),
            num_res_blocks=int(args.get("control_num_res_blocks", 4)),
        )
        if init_weight_path is not None:
            control_weight_path = os.path.join(init_weight_path, "trajectory_control_predictor.pt")
            if os.path.isfile(control_weight_path):
                control_predictor.load_state_dict(torch.load(control_weight_path, map_location="cpu"))
                logger.info(f"Initialized trajectory control predictor from {control_weight_path}")
            else:
                logger.info(
                    "No trajectory control predictor checkpoint found under init path; using random init."
                )
        control_predictor.train()

    # xformers
    if args.enable_xformers_memory_efficient_attention:
        if is_xformers_available():
            unet.enable_xformers_memory_efficient_attention()
        else:
            raise ValueError(
                "xformers is not available, please install it by running `pip install xformers`"
            )
        
    # DINOv3-ConvNeXt DISTS Loss
    from dinov3_gan.dinov3_convnext_dists import DINOv3ConvNeXtDISTS
    net_dv3d = DINOv3ConvNeXtDISTS(dinov3_convnext_size=args.dinov3_convnext_size)

    # DINOv3-ConvNeXt Discrminator
    from dinov3_gan.dinov3_convnext_disc import Dinov3ConvNeXtDiscriminator
    net_disc = Dinov3ConvNeXtDiscriminator(dinov3_convnext_size=args.dinov3_convnext_size, resolution=args.resolution)

    fixed_vae.to(device=accelerator.device)  
    lora_vae.to(device=accelerator.device) 
    unet.to(device=accelerator.device)
    if control_predictor is not None:
        control_predictor.to(device=accelerator.device)
    net_dv3d.to(device=accelerator.device)
    net_disc.to(device=accelerator.device)
    if args.gradient_checkpointing:
        lora_vae.enable_gradient_checkpointing()
        unet.enable_gradient_checkpointing()

    def unwrap_model(model):
        model = accelerator.unwrap_model(model)
        model = model._orig_mod if is_compiled_module(model) else model
        return model

    # Enable TF32 for faster training on Ampere GPUs,
    # cf https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices
    if args.allow_tf32 and torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True

    logger.info(
        f"Total vae_encoder training parameters: {sum([p.numel() for p in lora_vae.parameters() if p.requires_grad]) / 1000000} M"
    )
    logger.info(
        f"Total unet training parameters: {sum([p.numel() for p in unet.parameters() if p.requires_grad]) / 1000000} M"
    )
    if control_predictor is not None:
        logger.info(
            f"Total trajectory control training parameters: {sum([p.numel() for p in control_predictor.parameters() if p.requires_grad]) / 1000000} M"
        )
    logger.info(
        f"Total disc training parameters: {sum([p.numel() for p in net_disc.parameters() if p.requires_grad]) / 1000000} M"
    )
    sr_opt = list(filter(lambda p: p.requires_grad, lora_vae.parameters())) + list(filter(lambda p: p.requires_grad, unet.parameters()))
    if control_predictor is not None:
        sr_opt = sr_opt + list(filter(lambda p: p.requires_grad, control_predictor.parameters()))
    disc_opt = list(filter(lambda p: p.requires_grad, net_disc.parameters()))

    if args.use_8bit_adam:
        try:
            import bitsandbytes as bnb
        except ImportError:
            raise ImportError(
                "To use 8-bit Adam, please install the bitsandbytes library: `pip install bitsandbytes`."
            )

        optimizer_class = bnb.optim.AdamW8bit
    else:
        optimizer_class = torch.optim.AdamW

    optimizer_sr = optimizer_class(
        sr_opt,
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )

    optimizer_disc = optimizer_class(
        disc_opt,
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )

    train_dataset = PairedDataset(args.dataset_txt_or_dir_paths, args.resolution)
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.train_batch_size,
        shuffle=True,
        num_workers=args.dataloader_num_workers,
    )

    # Scheduler and math around the number of training steps.
    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(
        len(train_dataloader) / args.gradient_accumulation_steps
    )
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True

    lr_scheduler_sr = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer_sr,
        num_warmup_steps=args.lr_warmup_steps * accelerator.num_processes,
        num_training_steps=args.max_train_steps * accelerator.num_processes,
        num_cycles=args.lr_num_cycles,
        power=args.lr_power,
    )

    lr_scheduler_disc = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer_disc,
        num_warmup_steps=args.lr_warmup_steps * accelerator.num_processes,
        num_training_steps=args.max_train_steps * accelerator.num_processes,
        num_cycles=args.lr_num_cycles,
        power=args.lr_power,
    )

    if control_predictor is not None:
        (
            lora_vae,
            unet,
            control_predictor,
            net_disc,
            optimizer_sr,
            optimizer_disc,
            train_dataloader,
            lr_scheduler_sr,
            lr_scheduler_disc,
        ) = accelerator.prepare(
            lora_vae,
            unet,
            control_predictor,
            net_disc,
            optimizer_sr,
            optimizer_disc,
            train_dataloader,
            lr_scheduler_sr,
            lr_scheduler_disc,
        )
    else:
        (
            lora_vae,
            unet,
            net_disc,
            optimizer_sr,
            optimizer_disc,
            train_dataloader,
            lr_scheduler_sr,
            lr_scheduler_disc,
        ) = accelerator.prepare(
            lora_vae,
            unet,
            net_disc,
            optimizer_sr,
            optimizer_disc,
            train_dataloader,
            lr_scheduler_sr,
            lr_scheduler_disc,
        )
    net_disc.to(dtype=torch.float32)
    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(
        len(train_dataloader) / args.gradient_accumulation_steps
    )
    if overrode_max_train_steps:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    # Afterwards we recalculate our number of training epochs
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    # Train!
    total_batch_size = (
        args.train_batch_size
        * accelerator.num_processes
        * args.gradient_accumulation_steps
    )

    logger.info(f"***** Start training {args.model} *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num batches each epoch = {len(train_dataloader)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.train_batch_size}")
    logger.info(
        f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}"
    )
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")
    global_step = 0
    first_epoch = global_step // num_update_steps_per_epoch
    if init_weight_path is not None:
        logger.info(f"Initialized LoRA weights from checkpoint: {init_weight_path}")
        logger.info("Training starts from global_step=0 (no optimizer/scheduler state restore).")

    progress_bar = tqdm(
        range(0, args.max_train_steps),
        initial=global_step,
        desc="Steps",
        # Only show the progress bar once on each machine.
        disable=not accelerator.is_local_main_process,
    )
    ss_ratio_min = float(args.get("scheduled_sampling_ratio_min", 0.2))
    ss_ratio_max = float(args.get("scheduled_sampling_ratio_max", 0.9))
    ss_start_step = int(args.get("scheduled_sampling_start_step", 0))
    if ss_ratio_min > ss_ratio_max:
        raise ValueError("`scheduled_sampling_ratio_min` must be <= `scheduled_sampling_ratio_max`.")
    if ss_start_step < 0:
        raise ValueError("`scheduled_sampling_start_step` must be >= 0.")
    logger.info(
        f"Scheduled sampling config | ratio range=[{ss_ratio_min}, {ss_ratio_max}], start_step={ss_start_step}"
    )
    
    def get_alpha_coeffs(timestep):
        alpha_t = float(noise_scheduler.alphas_cumprod[timestep])
        sqrt_alphas_cumprod_t = math.sqrt(alpha_t)
        sqrt_one_minus_alphas_cumprod_t = math.sqrt(1 - alpha_t)
        return sqrt_alphas_cumprod_t, sqrt_one_minus_alphas_cumprod_t

    def normalize_sigma_map(sigma_map_raw):
        sigma_map = torch.sigmoid(sigma_map_raw)
        sigma_map = sigma_map / (sigma_map.mean(dim=(2, 3), keepdim=True) + 1e-6)
        sigma_map = torch.clamp(sigma_map, 0.0, 1.0)
        return sigma_map

    def sigma_map_to_pseudocolor(sigma_map):
        """Map [0,1] single-channel sigma map to a 3-channel jet-like pseudocolor image."""
        x = torch.clamp(sigma_map, 0.0, 1.0)
        r = torch.clamp(1.5 - torch.abs(4.0 * x - 3.0), 0.0, 1.0)
        g = torch.clamp(1.5 - torch.abs(4.0 * x - 2.0), 0.0, 1.0)
        b = torch.clamp(1.5 - torch.abs(4.0 * x - 1.0), 0.0, 1.0)
        return torch.cat([r, g, b], dim=1)

    def compute_control_features(control_latent):
        batch_size = control_latent.shape[0]
        device = control_latent.device
        dtype = control_latent.dtype
        has_control = control_predictor is not None
        sigma_map = torch.ones(
            (batch_size, 1, control_latent.shape[-2], control_latent.shape[-1]),
            device=device,
            dtype=dtype,
        )
        power = torch.ones((batch_size,), device=device, dtype=dtype)
        if has_control:
            control_outputs = control_predictor(control_latent.detach())
            sigma_map = normalize_sigma_map(control_outputs["sigma_map_raw"])
            if enable_power_modulation:
                timestep_lambda_raw = control_outputs["timestep_lambda_raw"].view(batch_size)
                power = power_min + (power_max - power_min) * torch.sigmoid(timestep_lambda_raw)
        return sigma_map, power, has_control

    def compute_bridge_controls(sigma_map_pred, power, has_control, sampled_sigma_value):
        batch_size = sigma_map_pred.shape[0]
        device = sigma_map_pred.device
        dtype = sigma_map_pred.dtype
        sigma_mix = torch.full(
            (batch_size, 1, 1, 1), float(sampled_sigma_value), device=device, dtype=dtype
        )
        if has_control and enable_power_modulation:
            sigma_eff = torch.pow(
                torch.full_like(power, float(sampled_sigma_value)),
                power,
            )
            sigma_mix = sigma_eff.view(batch_size, 1, 1, 1)
        sigma_map = (
            sigma_map_pred
            if (has_control and enable_sigma_map_modulation)
            else torch.ones_like(sigma_map_pred)
        )
        return sigma_mix, sigma_map

    def one_mid_timestep_pred_latent(input_latent, timestep, prompt_embeds_batch):
        model_pred = unet(input_latent, timestep, encoder_hidden_states=prompt_embeds_batch).sample
        sqrt_alphas_cumprod_t, sqrt_one_minus_alphas_cumprod_t = get_alpha_coeffs(timestep)
        denoised_latent = (input_latent - sqrt_one_minus_alphas_cumprod_t * model_pred) / sqrt_alphas_cumprod_t
        return denoised_latent

    def one_mid_timestep_pred(input_latent, timestep, prompt_embeds_batch):
        denoised_latent = one_mid_timestep_pred_latent(input_latent, timestep, prompt_embeds_batch)
        pred_img = (fixed_vae.decode(denoised_latent / fixed_vae.config.scaling_factor).sample).clamp(-1, 1)
        return pred_img, denoised_latent

    def full_step_pred(lq_latent, prompt_embeds_batch):
        pred_img = None
        pred_latent = None
        sigma_map_pred, power_pred, has_control = compute_control_features(lq_latent)
        for idx, timestep in enumerate(timestep_sequence):
            if idx == 0:
                step_input = lq_latent
            else:
                sigma = timestep_sigma_seq[timestep]
                step_sigma, step_sigma_map = compute_bridge_controls(
                    sigma_map_pred, power_pred, has_control, sigma
                )
                step_input = step_sigma * lq_latent + (1.0 - step_sigma) * pred_latent
                if args.bridge_sigma > 0:
                    # Deterministic visualization during training.
                    step_noise_scale = torch.sqrt(
                        torch.clamp(step_sigma * (1.0 - step_sigma), min=1e-12)
                    )
                    step_input = step_input + (
                        args.bridge_sigma
                        * step_sigma_map
                        * step_noise_scale
                        * torch.zeros_like(lq_latent)
                    )
            pred_img, pred_latent = one_mid_timestep_pred(step_input, timestep, prompt_embeds_batch)
        return pred_img

    def set_disc_requires_grad(flag):
        for p in disc_opt:
            p.requires_grad_(flag)

    models_to_accumulate = [lora_vae, unet, net_disc]
    if control_predictor is not None:
        models_to_accumulate.insert(2, control_predictor)
    power_residual_ema = torch.tensor(0.0, device=accelerator.device, dtype=torch.float32)
    power_ema_initialized = False
    power_mid = 0.5 * (power_min + power_max)
    power_half_range = 0.5 * (power_max - power_min)

    for epoch in range(first_epoch, args.num_train_epochs):
        for step, batch in enumerate(train_dataloader):
            with accelerator.accumulate(*models_to_accumulate):
                # Prepare data
                lq_img, hq_img = batch
                lq_img = lq_img.to(accelerator.device)
                hq_img = hq_img.to(accelerator.device)
                
                hq_latent = encode_images(hq_img, fixed_vae)
                pretrained_noisy_latent =  hq_latent 

                lq_latent = encode_images(lq_img, unwrap_model(lora_vae))

                # LRR Loss: Latent Representation Refinement Loss
                loss_LRR = F.mse_loss(pretrained_noisy_latent, lq_latent, reduction="mean") * args.lambda_LRR

                # Onestep prediction at sampled timeste
                if prompt_embeds.shape[0] == lq_latent.shape[0]:
                    prompt_embeds_batch = prompt_embeds
                else:
                    prompt_embeds_batch = prompt_embeds[:1].repeat(lq_latent.shape[0], 1, 1)

                # Warmup stage: force max timestep one-step training before scheduled sampling kicks in.
                if global_step < ss_start_step:
                    sampled_t = max_timestep
                    ss_ratio = 0.0
                    mixed_hq_latent = hq_latent
                else:
                    sampled_t = random.choice(timestep_sequence)
                    ss_ratio = random.uniform(ss_ratio_min, ss_ratio_max)
                    with torch.no_grad():
                        # Pseudo GT from one-step prediction at t_max.
                        pred_latent_teacher = one_mid_timestep_pred_latent(
                            lq_latent, max_timestep, prompt_embeds_batch
                        ).detach()
                    mixed_hq_latent = (1.0 - ss_ratio) * hq_latent + ss_ratio * pred_latent_teacher
                sampled_sigma = timestep_sigma_seq[sampled_t]

                sigma_map_pred, power_pred, has_control = compute_control_features(lq_latent)
                sigma_mix, sigma_map_mod = compute_bridge_controls(
                    sigma_map_pred, power_pred, has_control, sampled_sigma
                )
                bridge_noise = torch.randn_like(lq_latent)
                bridge_noise_scale = torch.sqrt(
                    torch.clamp(sigma_mix * (1.0 - sigma_mix), min=1e-12)
                )
                bridge_sample = (
                    sigma_mix * lq_latent
                    + (1.0 - sigma_mix) * mixed_hq_latent
                    + args.bridge_sigma
                    * sigma_map_mod
                    * bridge_noise_scale
                    * bridge_noise
                )

                pred_img, _ = one_mid_timestep_pred(bridge_sample, sampled_t, prompt_embeds_batch)

                # DINOv3-ConvNext DISTS Loss 
                loss_Dv3D = net_dv3d(pred_img, hq_img) * args.lambda_Dv3D

                # L1 Loss
                loss_L1 = F.l1_loss(pred_img, hq_img, reduction="mean") * args.lambda_L1

                # Generator Loss (SD)
                # Freeze the discriminator so the generator loss does not write its gradients.
                set_disc_requires_grad(False)
                loss_G = net_disc(pred_img.float(), for_G=True) * args.lambda_GAN

                loss_sigma_map = torch.zeros((), device=lq_latent.device, dtype=lq_latent.dtype)
                if enable_sigma_map_supervision:
                    sigma_map_target = torch.abs(lq_latent.detach() - hq_latent.detach()).mean(dim=1, keepdim=True)
                    sigma_map_target = sigma_map_target / (sigma_map_target.mean(dim=(2, 3), keepdim=True) + 1e-6)
                    sigma_map_target = torch.clamp(sigma_map_target, 0.0, 1.0)
                    loss_sigma_map = (
                        F.smooth_l1_loss(sigma_map_pred, sigma_map_target, reduction="mean")
                        * lambda_sigma_map_supervision
                    )

                loss_sigma_eff = torch.zeros((), device=lq_latent.device, dtype=lq_latent.dtype)
                power_target_mean = torch.zeros((), device=lq_latent.device, dtype=torch.float32)
                if enable_sigma_eff_supervision:
                    residual_strength = torch.abs(lq_latent.detach() - hq_latent.detach()).mean(dim=(1, 2, 3))
                    batch_residual_mean = residual_strength.mean().detach().float()
                    if not power_ema_initialized:
                        power_residual_ema = batch_residual_mean
                        power_ema_initialized = True
                    else:
                        power_residual_ema = (
                            power_ema_beta * power_residual_ema
                            + (1.0 - power_ema_beta) * batch_residual_mean
                        )

                    if global_step >= power_ema_warmup_steps:
                        residual_rel = torch.clamp(
                            residual_strength / (power_residual_ema + 1e-6),
                            min=0.0,
                            max=2.0,
                        )
                        power_target = torch.clamp(
                            power_mid + power_half_range * (residual_rel - 1.0),
                            min=power_min,
                            max=power_max,
                        )
                        power_target_mean = power_target.detach().float().mean()
                        loss_sigma_eff = (
                            F.smooth_l1_loss(power_pred, power_target, reduction="mean")
                            * lambda_sigma_eff_supervision
                        )

                total_G_loss = loss_LRR + loss_Dv3D + loss_L1 + loss_G + loss_sigma_map + loss_sigma_eff

                accelerator.backward(total_G_loss)
                set_disc_requires_grad(True)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(sr_opt, args.max_grad_norm)

                optimizer_sr.step()
                lr_scheduler_sr.step()
                optimizer_sr.zero_grad()
                
                fake_img = pred_img.detach()
                # Fake images
                loss_D_fake = net_disc(fake_img.float(), for_real=False) * args.lambda_GAN 
                # Real images
                accelerator.backward(loss_D_fake)
                
                loss_D_real = net_disc(hq_img.detach().float(), for_real=True) * args.lambda_GAN 
                accelerator.backward(loss_D_real)
          
                total_D_loss = loss_D_real + loss_D_fake 

                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(disc_opt, args.max_grad_norm)

                optimizer_disc.step()
                lr_scheduler_disc.step()
                optimizer_disc.zero_grad()

            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                if (
                    accelerator.is_main_process
                    and global_step % args.save_img_steps == 0
                ):
                    img_path = os.path.join(args.output_dir, f"img-{global_step}.jpg")
                    sigma_map_path = os.path.join(args.output_dir, f"sigma_map-{global_step}.jpg")
                    with torch.no_grad():
                        lq_latent_vis = encode_images(lq_img[:1], unwrap_model(lora_vae))
                        prompt_vis = prompt_embeds[:1]
                        pred_img_one, _ = one_mid_timestep_pred(lq_latent_vis, max_timestep, prompt_vis)
                        pred_img_full = full_step_pred(lq_latent_vis, prompt_vis)
                        sigma_map_vis = torch.clamp(sigma_map_pred[:1].detach().float(), 0.0, 1.0)
                        sigma_map_vis = F.interpolate(
                            sigma_map_vis,
                            size=(lq_img.shape[-2], lq_img.shape[-1]),
                            mode="bilinear",
                            align_corners=False,
                        )
                        sigma_map_vis = sigma_map_to_pseudocolor(sigma_map_vis)
                    save_imgs = (torch.stack([lq_img[0], pred_img_one[0], pred_img_full[0], hq_img[0]], dim=0) + 1) / 2
                    save_image(save_imgs.detach(), img_path)
                    save_image(sigma_map_vis, sigma_map_path)
                    logger.info(f"img-{global_step}.jpg saved! [lq, one_step@max_t, full_step, hq]")
                    logger.info(f"sigma_map-{global_step}.jpg saved!")

                progress_bar.update(1)
                global_step += 1

                if (
                    accelerator.is_main_process
                    or accelerator.distributed_type == DistributedType.DEEPSPEED
                ):
                    if global_step % args.checkpointing_steps == 0:
                        weight_path = os.path.join(
                            args.output_dir, f"weight-{global_step}"
                        )
                        os.makedirs(weight_path, exist_ok=True)
                        unwrap_model(unet).save_pretrained(weight_path)
                        unwrap_model(lora_vae).encoder.save_pretrained(weight_path)
                        if control_predictor is not None:
                            torch.save(
                                unwrap_model(control_predictor).state_dict(),
                                os.path.join(weight_path, "trajectory_control_predictor.pt"),
                            )
                        logger.info(f"Saved weight to {weight_path}")

            logs = {
                "loss_LRR": loss_LRR.detach().item(),
                "loss_D_fake": loss_D_fake.detach().item(),
                "loss_D_real": loss_D_real.detach().item(),
                "loss_Dv3D": loss_Dv3D.detach().item(),
                "loss_L1": loss_L1.detach().item(),
                "loss_sigma_map": loss_sigma_map.detach().item(),
                "loss_sigma_eff": loss_sigma_eff.detach().item(),
                "sampled_t": sampled_t,
                "sampled_sigma": sampled_sigma,
                "sigma_eff_mean": sigma_mix.detach().mean().item(),
                "power_mean": power_pred.detach().mean().item(),
                "power_target_mean": power_target_mean.detach().item(),
                "power_residual_ema": power_residual_ema.detach().item() if power_ema_initialized else 0.0,
                "ss_ratio": ss_ratio,
                "lr": lr_scheduler_sr.get_last_lr()[0],
            }
            progress_bar.set_postfix(**logs)
            accelerator.log(logs, step=global_step)

            if global_step >= args.max_train_steps:
                break

    # Save the lora layers
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        weight_path = os.path.join(args.output_dir, f"weight-{global_step}")
        os.makedirs(weight_path, exist_ok=True)
        unwrap_model(unet).save_pretrained(weight_path)
        unwrap_model(lora_vae).encoder.save_pretrained(weight_path)
        if control_predictor is not None:
            torch.save(
                unwrap_model(control_predictor).state_dict(),
                os.path.join(weight_path, "trajectory_control_predictor.pt"),
            )
        logger.info(f"Saved weight to {weight_path}")

    accelerator.end_training()


if __name__ == "__main__":
    main()
