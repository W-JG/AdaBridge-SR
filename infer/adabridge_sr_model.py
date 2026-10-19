import os
import time

import torch
from diffusers import AutoencoderKL, DDPMScheduler, UNet2DConditionModel
from peft import PeftModel

from adabridge.trajectory_control import TrajectoryControlPredictor


class AdaBridgeSR(torch.nn.Module):
    def __init__(
        self,
        sd_path,
        lora_path,
        timestep_sequence,
        bridge_sigma,
        device,
        weight_dtype,
        enable_trajectory_control=False,
        enable_power_modulation=False,
        enable_sigma_map_modulation=False,
        power_min=0.5,
        power_max=1.5,
        control_base_channels=64,
        control_num_res_blocks=4,
    ):
        super().__init__()
        vae = AutoencoderKL.from_pretrained(sd_path, subfolder="vae")
        unet = UNet2DConditionModel.from_pretrained(sd_path, subfolder="unet")

        vae.encoder = PeftModel.from_pretrained(
            vae.encoder, os.path.join(lora_path, "vae_encoder_lora_adapter")
        )
        unet = PeftModel.from_pretrained(
            unet, os.path.join(lora_path, "unet_lora_adapter")
        )
        vae.encoder.merge_and_unload()
        unet.merge_and_unload()

        self.scheduler = DDPMScheduler.from_pretrained(sd_path, subfolder="scheduler")
        self.timestep_sequence = [int(t) for t in timestep_sequence]
        if len(self.timestep_sequence) == 0:
            raise ValueError("`timestep_sequence` must contain at least one timestep.")
        self.max_timestep = max(self.timestep_sequence + [0])
        if self.max_timestep <= 0:
            raise ValueError("`max(timestep_sequence)` must be > 0 for linear sigma mapping.")
        self.timestep_sigma_seq = {
            int(t): float(t) / float(self.max_timestep) for t in set(self.timestep_sequence + [0])
        }

        self.enable_trajectory_control = bool(enable_trajectory_control)
        self.enable_power_modulation = bool(enable_power_modulation)
        self.enable_sigma_map_modulation = bool(enable_sigma_map_modulation)
        self.power_min = float(power_min)
        self.power_max = float(power_max)
        if self.power_min > self.power_max:
            raise ValueError("`power_min` must be <= `power_max`.")

        self.bridge_sigma = bridge_sigma
        self.vae = vae.to(device=device, dtype=weight_dtype)
        self.unet = unet.to(device=device, dtype=weight_dtype)
        self.device = device
        self.control_predictor = None
        if self.enable_trajectory_control:
            self.control_predictor = TrajectoryControlPredictor(
                in_channels=4,
                base_channels=int(control_base_channels),
                num_res_blocks=int(control_num_res_blocks),
            )
            control_weight_path = os.path.join(lora_path, "trajectory_control_predictor.pt")
            if not os.path.isfile(control_weight_path):
                raise ValueError(
                    f"`enable_trajectory_control=True` but no predictor weight found: {control_weight_path}"
                )
            self.control_predictor.load_state_dict(torch.load(control_weight_path, map_location="cpu"))
            self.control_predictor = self.control_predictor.to(device=device, dtype=weight_dtype)

        self.vae.eval()
        self.unet.eval()
        if self.control_predictor is not None:
            self.control_predictor.eval()

    def _gaussian_weights(self, tile_width, tile_height, nbatches):
        from numpy import exp, pi, sqrt
        import numpy as np

        latent_width = tile_width
        latent_height = tile_height

        var = 0.01
        midpoint = (latent_width - 1) / 2
        x_probs = [
            exp(-(x - midpoint) * (x - midpoint) / (latent_width * latent_width) / (2 * var))
            / sqrt(2 * pi * var)
            for x in range(latent_width)
        ]
        midpoint = latent_height / 2
        y_probs = [
            exp(-(y - midpoint) * (y - midpoint) / (latent_height * latent_height) / (2 * var))
            / sqrt(2 * pi * var)
            for y in range(latent_height)
        ]

        weights = np.outer(y_probs, x_probs)
        return torch.tile(
            torch.tensor(weights, device=self.device),
            (nbatches, self.unet.config.in_channels, 1, 1),
        )

    def _alpha_t(self, timestep):
        return self.scheduler.alphas_cumprod[timestep]

    def _sigma_from_t(self, timestep):
        return self.timestep_sigma_seq[int(timestep)]

    def _normalize_sigma_map(self, sigma_map_raw):
        sigma_map = torch.sigmoid(sigma_map_raw)
        sigma_map = sigma_map / (sigma_map.mean(dim=(2, 3), keepdim=True) + 1e-6)
        sigma_map = torch.clamp(sigma_map, 0.0, 1.0)
        return sigma_map

    def _compute_control_features(self, latent):
        batch_size = latent.shape[0]
        has_control = self.control_predictor is not None
        sigma_map = torch.ones(
            (batch_size, 1, latent.shape[-2], latent.shape[-1]),
            device=latent.device,
            dtype=latent.dtype,
        )
        power = torch.ones((batch_size,), device=latent.device, dtype=latent.dtype)
        if has_control:
            control_outputs = self.control_predictor(latent)
            sigma_map = self._normalize_sigma_map(control_outputs["sigma_map_raw"])
            if self.enable_power_modulation:
                timestep_lambda_raw = control_outputs["timestep_lambda_raw"].view(batch_size)
                power = self.power_min + (self.power_max - self.power_min) * torch.sigmoid(timestep_lambda_raw)
        return sigma_map, power, has_control

    def _compute_bridge_controls(self, sigma_map_pred, power, has_control, sigma_value):
        batch_size = sigma_map_pred.shape[0]
        sigma_mix = torch.full(
            (batch_size, 1, 1, 1),
            float(sigma_value),
            device=sigma_map_pred.device,
            dtype=sigma_map_pred.dtype,
        )

        if has_control and self.enable_power_modulation:
            sigma_eff = torch.pow(torch.full_like(power, float(sigma_value)), power)
            sigma_mix = sigma_eff.view(batch_size, 1, 1, 1)

        sigma_map = (
            sigma_map_pred
            if (has_control and self.enable_sigma_map_modulation)
            else torch.ones_like(sigma_map_pred)
        )
        return sigma_mix, sigma_map

    def _select_infer_timesteps(self, num_infer_steps):
        total_steps = len(self.timestep_sequence)
        num_infer_steps = int(num_infer_steps)
        if num_infer_steps < 1 or num_infer_steps > total_steps:
            raise ValueError(
                f"`num_infer_steps` must be in [1, {total_steps}], got {num_infer_steps}."
            )

        # Timestep selection for a sequence [t4,t3,t2,t1]:
        # [t4,t3,t2,t1] -> 1-step:[t4], 2-step:[t4,t1], 3-step:[t4,t2,t1], 4-step:[t4,t3,t2,t1]
        if num_infer_steps == 1:
            return [self.timestep_sequence[0]]
        suffix_len = num_infer_steps - 1
        return [self.timestep_sequence[0]] + self.timestep_sequence[-suffix_len:]

    def _forward_no_tile(self, input_latent, prompt_embeds, timestep):
        model_pred = self.unet(
            input_latent.to(self.unet.dtype),
            timestep,
            encoder_hidden_states=prompt_embeds,
        ).sample
        alpha_t = self._alpha_t(timestep)
        denoised_latent = (
            input_latent - (1 - alpha_t).sqrt() * model_pred.to(self.vae.dtype)
        ) / alpha_t.sqrt()
        pred_img = (
            self.vae.decode(denoised_latent / self.vae.config.scaling_factor).sample
        ).clamp(-1, 1)
        return pred_img, denoised_latent

    def _forward_tile(self, input_latent, prompt_embeds, timestep, tile_size, tile_overlap):
        _, _, h, w = input_latent.shape
        tile_size = min(tile_size, min(h, w))
        tile_weights = self._gaussian_weights(tile_size, tile_size, 1)

        grid_rows = 0
        cur_x = 0
        while cur_x < input_latent.size(-1):
            cur_x = max(grid_rows * tile_size - tile_overlap * grid_rows, 0) + tile_size
            grid_rows += 1

        grid_cols = 0
        cur_y = 0
        while cur_y < input_latent.size(-2):
            cur_y = max(grid_cols * tile_size - tile_overlap * grid_cols, 0) + tile_size
            grid_cols += 1

        input_list = []
        noise_preds = []
        for row in range(grid_rows):
            for col in range(grid_cols):
                if col < grid_cols - 1 or row < grid_rows - 1:
                    ofs_x = max(row * tile_size - tile_overlap * row, 0)
                    ofs_y = max(col * tile_size - tile_overlap * col, 0)
                if row == grid_rows - 1:
                    ofs_x = w - tile_size
                if col == grid_cols - 1:
                    ofs_y = h - tile_size

                input_start_x = ofs_x
                input_end_x = ofs_x + tile_size
                input_start_y = ofs_y
                input_end_y = ofs_y + tile_size

                input_tile = input_latent[:, :, input_start_y:input_end_y, input_start_x:input_end_x]
                input_list.append(input_tile)

                if len(input_list) == 1 or col == grid_cols - 1:
                    input_list_t = torch.cat(input_list, dim=0)
                    model_out = self.unet(
                        input_list_t.to(self.unet.dtype),
                        timestep,
                        encoder_hidden_states=prompt_embeds.to(self.unet.dtype),
                    ).sample
                    input_list = []
                noise_preds.append(model_out)

        noise_pred = torch.zeros(input_latent.shape, device=input_latent.device)
        contributors = torch.zeros(input_latent.shape, device=input_latent.device)
        for row in range(grid_rows):
            for col in range(grid_cols):
                if col < grid_cols - 1 or row < grid_rows - 1:
                    ofs_x = max(row * tile_size - tile_overlap * row, 0)
                    ofs_y = max(col * tile_size - tile_overlap * col, 0)
                if row == grid_rows - 1:
                    ofs_x = w - tile_size
                if col == grid_cols - 1:
                    ofs_y = h - tile_size

                input_start_x = ofs_x
                input_end_x = ofs_x + tile_size
                input_start_y = ofs_y
                input_end_y = ofs_y + tile_size

                noise_pred[:, :, input_start_y:input_end_y, input_start_x:input_end_x] += (
                    noise_preds[row * grid_cols + col] * tile_weights
                )
                contributors[:, :, input_start_y:input_end_y, input_start_x:input_end_x] += tile_weights

        model_pred = noise_pred / contributors
        alpha_t = self._alpha_t(timestep)
        denoised_latent = (
            input_latent - (1 - alpha_t).sqrt() * model_pred.to(self.vae.dtype)
        ) / alpha_t.sqrt()
        pred_image = (
            self.vae.decode(denoised_latent / self.vae.config.scaling_factor).sample
        ).clamp(-1, 1)
        return pred_image, denoised_latent

    def _step(self, input_latent, prompt_embeds, timestep, tile_size, tile_overlap):
        _, _, h, w = input_latent.size()
        if h * w <= tile_size * tile_size:
            return self._forward_no_tile(input_latent, prompt_embeds, timestep)
        return self._forward_tile(input_latent, prompt_embeds, timestep, tile_size, tile_overlap)

    def forward(self, lq_img, prompt_embeds, tile_size, tile_overlap, num_infer_steps=1):
        torch.cuda.synchronize()
        start_time = time.time()

        lq_latent = self.vae.encode(lq_img.to(self.vae.dtype)).latent_dist.sample()
        lq_latent = lq_latent * self.vae.config.scaling_factor
        infer_timesteps = self._select_infer_timesteps(num_infer_steps)
        pred_latent = None
        pred_img = None
        sigma_map_pred, power_pred, has_control = self._compute_control_features(lq_latent)
        for idx, timestep in enumerate(infer_timesteps):
            if idx == 0:
                step_input = lq_latent
            else:
                sigma = self._sigma_from_t(timestep)
                step_noise = torch.randn_like(lq_latent)
                sigma_mix, sigma_map = self._compute_bridge_controls(
                    sigma_map_pred, power_pred, has_control, sigma
                )
                sigma_noise_scale = torch.sqrt(torch.clamp(sigma_mix * (1.0 - sigma_mix), min=0.0))
                step_input = (
                    sigma_mix * lq_latent
                    + (1.0 - sigma_mix) * pred_latent
                    + self.bridge_sigma * sigma_map * sigma_noise_scale * step_noise
                )
            pred_img, pred_latent = self._step(
                step_input,
                prompt_embeds,
                timestep,
                tile_size,
                tile_overlap,
            )

        torch.cuda.synchronize()
        elapsed = time.time() - start_time
        return pred_img, elapsed
