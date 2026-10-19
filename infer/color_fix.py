"""Color correction for AdaBridge-SR outputs."""
import torch
from PIL import Image
from torchvision.transforms import functional as TF


def _channel_stats(x: torch.Tensor, eps: float = 1e-5):
    """Per-channel mean and standard deviation of a (B, C, H, W) tensor."""
    flat = x.flatten(2)
    mean = flat.mean(dim=2)[..., None, None]
    std = (flat.var(dim=2) + eps).sqrt()[..., None, None]
    return mean, std


def adain_color_fix(target: Image.Image, source: Image.Image) -> Image.Image:
    """Match the per-channel mean and standard deviation of `target` to `source`."""
    target_t = TF.to_tensor(target).unsqueeze(0)
    source_t = TF.to_tensor(source).unsqueeze(0)
    target_mean, target_std = _channel_stats(target_t)
    source_mean, source_std = _channel_stats(source_t)
    result = (target_t - target_mean) / target_std * source_std + source_mean
    return TF.to_pil_image(result.squeeze(0).clamp(0.0, 1.0))
