import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        groups = min(8, channels)
        self.norm1 = nn.GroupNorm(groups, channels)
        self.act1 = nn.SiLU()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.norm2 = nn.GroupNorm(groups, channels)
        self.act2 = nn.SiLU()
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)

    def forward(self, x):
        residual = x
        x = self.conv1(self.act1(self.norm1(x)))
        x = self.conv2(self.act2(self.norm2(x)))
        return x + residual


class TrajectoryControlPredictor(nn.Module):
    """Shared backbone with two heads for timestep/sigma-map prediction."""

    def __init__(self, in_channels=4, base_channels=64, num_res_blocks=4):
        super().__init__()

        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, base_channels, kernel_size=3, padding=1),
            nn.GroupNorm(min(8, base_channels), base_channels),
            nn.SiLU(),
        )
        self.blocks = nn.Sequential(*[ResidualBlock(base_channels) for _ in range(num_res_blocks)])

        self.timestep_head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(base_channels, base_channels),
            nn.SiLU(),
            nn.Linear(base_channels, 1),
        )
        self.sigma_map_head = nn.Conv2d(base_channels, 1, kernel_size=1)

    def forward(self, latent):
        feat = self.blocks(self.stem(latent))
        timestep_param = self.timestep_head(feat)
        sigma_map_raw = self.sigma_map_head(feat)
        if sigma_map_raw.shape[-2:] != latent.shape[-2:]:
            sigma_map_raw = F.interpolate(sigma_map_raw, size=latent.shape[-2:], mode="bilinear", align_corners=False)
        outputs = {"sigma_map_raw": sigma_map_raw}
        
        outputs["timestep_lambda_raw"] = timestep_param.squeeze(-1)
        return outputs
