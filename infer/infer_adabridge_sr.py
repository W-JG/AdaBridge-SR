import sys
import os
import argparse
import random
from pathlib import Path
from PIL import Image
import torch
from torchvision import transforms
import torchvision.transforms.functional as F
from tqdm import tqdm
from omegaconf import OmegaConf

SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))

from infer.color_fix import adain_color_fix
import glob
from infer.adabridge_sr_model import AdaBridgeSR
from diffusers.training_utils import free_memory
from transformers import AutoTokenizer, CLIPTextModel


def _resolve_latest_weight_dir(output_dir):
    if not output_dir or not os.path.isdir(output_dir):
        return None
    dirs = [d for d in os.listdir(output_dir) if d.startswith("weight-")]
    dirs = sorted(
        dirs,
        key=lambda x: int(x.split("-")[1]) if x.split("-")[1].isdigit() else -1,
    )
    if len(dirs) == 0:
        return None
    return os.path.join(output_dir, dirs[-1])


def _parse_unknown_args_to_cfg(unknown_args):
    dotlist = []
    idx = 0
    while idx < len(unknown_args):
        token = unknown_args[idx]
        if not token.startswith("--"):
            raise ValueError(f"Unexpected token in unknown args: {token}")
        item = token[2:]
        if "=" in item:
            key, value = item.split("=", 1)
        else:
            if item.startswith("no-"):
                key = item[3:]
                value = "false"
            elif idx + 1 < len(unknown_args) and not unknown_args[idx + 1].startswith("--"):
                key = item
                value = unknown_args[idx + 1]
                idx += 1
            else:
                key = item
                value = "true"
        key = key.replace("-", "_")
        dotlist.append(f"{key}={value}")
        idx += 1
    if len(dotlist) == 0:
        return OmegaConf.create()
    return OmegaConf.from_dotlist(dotlist)


def set_seed(seed):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except ImportError:
        pass

    # Keep CUDA kernel selection deterministic for reproducible inference.
    if torch.backends.cudnn.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def main(args):
    set_seed(args.seed)
    max_timestep = max(args.timestep_sequence + [0])
    if max_timestep <= 0:
        raise ValueError("`max(timestep_sequence)` must be > 0 for linear sigma mapping.")

    tokenizer = AutoTokenizer.from_pretrained(args.sd_path, subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(
        args.sd_path, subfolder="text_encoder"
    ).to(args.device, dtype=args.weight_dtype)

    def encode_prompt(prompt_batch):
        with torch.no_grad():
            prompt_embeds = [
                text_encoder(
                    tokenizer(
                        caption,
                        max_length=tokenizer.model_max_length,
                        padding="max_length",
                        truncation=True,
                        return_tensors="pt",
                    ).input_ids.to(device=text_encoder.device)
                )[0]
                for caption in prompt_batch
            ]
        return torch.concat(prompt_embeds, dim=0)

    prompt_embeds = encode_prompt([args.prompt])
    del tokenizer
    del text_encoder
    free_memory()

    net_sr = AdaBridgeSR(
        sd_path=args.sd_path,
        lora_path=args.lora_path,
        timestep_sequence=args.timestep_sequence,
        bridge_sigma=args.bridge_sigma,
        device=args.device,
        weight_dtype=args.weight_dtype,
        enable_trajectory_control=args.enable_trajectory_control,
        enable_power_modulation=args.enable_power_modulation,
        enable_sigma_map_modulation=args.enable_sigma_map_modulation,
        power_min=args.power_min,
        power_max=args.power_max,
        control_base_channels=args.control_base_channels,
        control_num_res_blocks=args.control_num_res_blocks,
    )

    if ".txt" in args.input_image:
        with open(args.input_image, "r") as f:
            image_names = [l.strip() for l in f.readlines()]
    else:
        if os.path.isdir(args.input_image):
            image_names = sorted(
                glob.glob(f"{args.input_image}/*.png")
                + glob.glob(f"{args.input_image}/*.jpg")
                + glob.glob(f"{args.input_image}/*.jpeg")
            )
        else:
            image_names = [args.input_image]
    if len(image_names) == 0:
        raise ValueError(f"No input images found: {args.input_image}")

    os.makedirs(args.output_dir, exist_ok=True)
    print(f"There are {len(image_names)} images.")

    total_time = 0
    for image_name in tqdm(image_names):
        input_image = Image.open(image_name).convert("RGB")
        ori_width, ori_height = input_image.size
        rscale = args.upscale
        resize_flag = False

        if ori_width < args.process_size // rscale or ori_height < args.process_size // rscale:
            scale = (args.process_size // rscale) / min(ori_width, ori_height)
            input_image = input_image.resize((int(scale * ori_width), int(scale * ori_height)))
            resize_flag = True

        input_image = input_image.resize((input_image.size[0] * rscale, input_image.size[1] * rscale))
        new_width = input_image.width - input_image.width % 8
        new_height = input_image.height - input_image.height % 8
        input_image = input_image.resize((new_width, new_height), Image.LANCZOS)
        bname = os.path.basename(image_name).split(".")[0] + ".png"

        tile_size = args.process_size // 8
        tile_overlap = tile_size // 2

        with torch.no_grad():
            lq_img = F.to_tensor(input_image).unsqueeze(0).to(device=args.device, dtype=args.weight_dtype) * 2 - 1
            output_image, infer_time = net_sr(
                lq_img,
                prompt_embeds,
                tile_size,
                tile_overlap,
                num_infer_steps=args.num_infer_steps,
            )
            total_time += infer_time

        output_image = output_image * 0.5 + 0.5
        output_image = torch.clip(output_image, 0, 1).float()
        output_pil = transforms.ToPILImage()(output_image[0].cpu())

        if args.align_method == "adain":
            output_pil = adain_color_fix(target=output_pil, source=input_image)

        if resize_flag:
            output_pil = output_pil.resize((int(args.upscale * ori_width), int(args.upscale * ori_height)))
        output_pil.save(os.path.join(args.output_dir, bname))

    print(f"Average inference time: {total_time / len(image_names)}s")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AdaBridge-SR inference")

    parser.add_argument("--input_image", type=str, required=True,
                        help="Path to input image or directory containing images")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Output directory for results")
    parser.add_argument("--train_cfg", type=str, required=True,
                        help="Training cfg.yml path. Most inference options are read from this file.")

    known_args, unknown_args = parser.parse_known_args()
    train_cfg = OmegaConf.load(known_args.train_cfg)
    override_cfg = _parse_unknown_args_to_cfg(unknown_args)
    cfg = OmegaConf.merge(train_cfg, override_cfg)

    sd_path = cfg.get("sd_path", None)
    if sd_path is None:
        raise ValueError("`sd_path` must be provided in train cfg or unknown args.")

    lora_path = cfg.get("lora_path", None)
    if lora_path is None:
        lora_path = _resolve_latest_weight_dir(cfg.get("output_dir", None))
    if lora_path is None:
        raise ValueError(
            "`lora_path` is missing and no `weight-*` is found under `output_dir` in train cfg."
        )

    timestep_sequence = cfg.get("timestep_sequence", None)
    if timestep_sequence is None:
        raise ValueError("`timestep_sequence` must be provided in train cfg or unknown args.")

    align_method = cfg.get("align_method", "adain")
    if align_method not in ["adain", "nofix"]:
        raise ValueError(f"Unsupported `align_method`: {align_method}")

    weight_dtype_key = str(cfg.get("weight_dtype", "bf16"))
    if weight_dtype_key not in ["fp32", "fp16", "bf16"]:
        raise ValueError(f"Unsupported `weight_dtype`: {weight_dtype_key}")

    args = argparse.Namespace(
        input_image=known_args.input_image,
        output_dir=known_args.output_dir,
        sd_path=str(sd_path),
        lora_path=str(Path(lora_path)),
        device=str(cfg.get("device", "cuda:0")),
        process_size=int(cfg.get("process_size", cfg.get("resolution", 512))),
        upscale=int(cfg.get("upscale", 4)),
        align_method=align_method,
        weight_dtype=weight_dtype_key,
        prompt=str(cfg.get("prompt", cfg.get("fixed_prompt", ""))),
        timestep_sequence=[int(t) for t in timestep_sequence],
        bridge_sigma=float(cfg.get("bridge_sigma", 0.0)),
        num_infer_steps=int(cfg.get("num_infer_steps", 1)),
        enable_trajectory_control=bool(cfg.get("enable_trajectory_control", False)),
        enable_power_modulation=bool(cfg.get("enable_power_modulation", False)),
        enable_sigma_map_modulation=bool(cfg.get("enable_sigma_map_modulation", False)),
        power_min=float(cfg.get("power_min", 0.5)),
        power_max=float(cfg.get("power_max", 1.5)),
        control_base_channels=int(cfg.get("control_base_channels", 64)),
        control_num_res_blocks=int(cfg.get("control_num_res_blocks", 4)),
        seed=int(cfg.get("seed", 42)),
    )

    args.weight_dtype = {
        "fp32": torch.float32,
        "fp16": torch.float16,
        "bf16": torch.bfloat16,
    }[args.weight_dtype]

    main(args)
