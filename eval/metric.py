import os
import sys
import glob
import argparse
import logging
from datetime import datetime
import time
import csv
import fcntl

import cv2
import torch
import pyiqa


# -----------------------------
# Metric configs (order matters for CSV columns)
# -----------------------------
METRIC_CONFIGS = {
    # --- Full-Reference (FR) ---
    "PSNR": {
        "type": "FR",
        "net_name": "psnr",
        "opts": {"test_y_channel": True, "color_space": "ycbcr"},
    },
    "SSIM": {
        "type": "FR",
        "net_name": "ssim",
        "opts": {"test_y_channel": True, "color_space": "ycbcr"},
    },
    "LPIPS": {"type": "FR", "net_name": "lpips", "opts": {}},
    "DISTS": {"type": "FR", "net_name": "dists", "opts": {}},
    # --- No-Reference (NR) ---
    "CLIPIQA": {"type": "NR", "net_name": "clipiqa", "opts": {}},
    "NIQE": {"type": "NR", "net_name": "niqe", "opts": {}},
    "MUSIQ": {"type": "NR", "net_name": "musiq", "opts": {}},
    "MANIQA": {
        "type": "NR",
        "net_name": "maniqa-pipal",
        "opts": {},
    },
    "CLIPIQA+": {"type": "NR", "net_name": "clipiqa+", "opts": {}},
    "topiq_nr": {"type": "NR", "net_name": "topiq_nr", "opts": {}},
     
}

ALL_METRIC_NAMES_FIXED = list(METRIC_CONFIGS.keys())
PLACEHOLDER_GTS = {"none", "no_gt", "null", "nan"}


def img2tensor(imgs, bgr2rgb=True, float32=True):
    """Numpy array to tensor."""

    def _totensor(img, bgr2rgb, float32):
        if img.shape[2] == 3 and bgr2rgb:
            if img.dtype == "float64":
                img = img.astype("float32")
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = torch.from_numpy(img.transpose(2, 0, 1))
        if float32:
            img = img.float()
        return img

    if isinstance(imgs, list):
        return [_totensor(img, bgr2rgb, float32) for img in imgs]
    else:
        return _totensor(imgs, bgr2rgb, float32)


def get_timestamp():
    return datetime.now().strftime("%y%m%d-%H%M%S")


def setup_logger(logger_name, log_dir_for_files, phase, level=logging.INFO, screen=True, tofile=True):
    logger = logging.getLogger(logger_name)
    formatter = logging.Formatter(
        fmt="%(asctime)s.%(msecs)03d - %(levelname)s: %(message)s",
        datefmt="%y-%m-%d %H:%M:%S",
    )
    logger.setLevel(level)
    logger.propagate = False

    if logger.handlers:
        logger.handlers.clear()

    log_file = None
    if tofile:
        os.makedirs(log_dir_for_files, exist_ok=True)
        log_file = os.path.join(log_dir_for_files, f"{phase}_{get_timestamp()}.log")
        fh = logging.FileHandler(log_file, mode="w")
        fh.setFormatter(formatter)
        logger.addHandler(fh)

    if screen:
        sh = logging.StreamHandler()
        sh.setFormatter(formatter)
        logger.addHandler(sh)

    return log_file


def dict2str(opt, indent=1):
    msg = ""
    for k, v in opt.items():
        if isinstance(v, dict):
            msg += " " * (indent * 2) + f"{k}:[\n"
            msg += dict2str(v, indent + 1)
            msg += " " * (indent * 2) + "]\n"
        else:
            msg += " " * (indent * 2) + f"{k}: {v}\n"
    return msg


def is_placeholder_gt(s: str) -> bool:
    if s is None:
        return True
    return str(s).strip().lower() in PLACEHOLDER_GTS


def list_images(dir_path: str):
    exts = ["png", "jpg", "jpeg", "bmp", "webp"]
    paths = []
    for ext in exts:
        paths += glob.glob(os.path.join(dir_path, f"*.{ext}"))
        paths += glob.glob(os.path.join(dir_path, f"*.{ext.upper()}"))
    return sorted(set(paths))


def build_stem_map(img_paths):
    stem_map = {}
    dup = 0
    for p in img_paths:
        base = os.path.basename(p)
        stem, _ = os.path.splitext(base)
        if stem in stem_map:
            dup += 1
            continue
        stem_map[stem] = p
    return stem_map, dup


def maybe_cuda_sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize()


def append_avg_csv_row(avg_csv_path, header, row):
    """Append one row with process-safe file locking; writes header if file is empty."""
    with open(avg_csv_path, "a+", newline="") as f:
        fcntl.flock(f.fileno(), fcntl.LOCK_EX)
        f.seek(0, os.SEEK_END)
        need_header = f.tell() == 0
        w = csv.writer(f)
        if need_header:
            w.writerow(header)
        w.writerow(row)
        f.flush()
        os.fsync(f.fileno())
        fcntl.flock(f.fileno(), fcntl.LOCK_UN)


def main():
    parser = argparse.ArgumentParser(description="Robust IQA Script (FR+NR, optional GT)")

    parser.add_argument("--inp_imgs", nargs="+", required=True, help="Input (SR) image directories.")
    parser.add_argument("--gt_imgs", nargs="+", required=True, help="GT directories or placeholder None/no_gt.")
    parser.add_argument("--log", type=str, required=True, help="Root output dir.")
    parser.add_argument("--model_name", type=str, default="MODEL", help="Model name (used in filenames).")

    # optional explicit dataset names
    parser.add_argument(
        "--dataset_names",
        nargs="+",
        default=None,
        help="Optional dataset names aligned with --inp_imgs (same count). If omitted, use basename of SR dir.",
    )

    args = parser.parse_args()

    if len(args.inp_imgs) != len(args.gt_imgs):
        print(f"[ERROR] mismatch: {len(args.inp_imgs)} inp dirs vs {len(args.gt_imgs)} gt dirs.")
        sys.exit(1)

    if args.dataset_names is not None and len(args.dataset_names) != len(args.inp_imgs):
        print(f"[ERROR] mismatch: {len(args.dataset_names)} dataset_names vs {len(args.inp_imgs)} inp dirs.")
        sys.exit(1)

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    # Root folders
    root_out = args.log
    logs_dir = os.path.join(root_out, "logs")          # (1) all log files go here
    details_root = os.path.join(root_out, "details")   # (2) all detail csv go here
    os.makedirs(root_out, exist_ok=True)
    os.makedirs(logs_dir, exist_ok=True)
    os.makedirs(details_root, exist_ok=True)

    model_name = args.model_name  

    # Setup logger -> ./log/logs/
    log_file_path = setup_logger("base", logs_dir, f"test_{model_name}", level=logging.INFO, screen=True, tofile=True)
    logger = logging.getLogger("base")

    logger.info("===== Configuration =====")
    logger.info(dict2str(vars(args)))
    logger.info(f"Device: {device}")
    logger.info(f"Log file: {log_file_path}")
    logger.info(f"Root out: {root_out}")

    # Pre-scan for any valid GT to decide whether to load FR
    any_valid_gt_dataset = False
    for gt in args.gt_imgs:
        if is_placeholder_gt(gt):
            continue
        if os.path.exists(gt):
            any_valid_gt_dataset = True
            break

    # Initialize metrics
    logger.info("Initializing IQA metrics...")
    fr_metric_names = []
    nr_metric_names = []
    metrics_models = {}

    for name, config in METRIC_CONFIGS.items():
        mtype = config["type"]
        if mtype == "FR" and not any_valid_gt_dataset:
            continue

        try:
            logger.info(f"Loading {name} ({config['net_name']})...")
            metric_model = pyiqa.create_metric(config["net_name"], device=device, **config.get("opts", {})).to(device)
            metric_model.eval()
            metrics_models[name] = metric_model

            if mtype == "FR":
                fr_metric_names.append(name)
            else:
                nr_metric_names.append(name)
        except Exception as e:
            logger.error(f"Failed to init metric [{name}]: {e}")

    logger.info(f"Initialized FR Metrics: {fr_metric_names}")
    logger.info(f"Initialized NR Metrics: {nr_metric_names}")
    logger.info("IQA metrics initialized.\n")

    logger.info("\n===== Starting Evaluation =====\n")

    run_ts = get_timestamp()

    # Evaluate each dataset
    for dir_idx, sr_dir in enumerate(args.inp_imgs):
        raw_gt_input = args.gt_imgs[dir_idx]

        dataset_name = (
            args.dataset_names[dir_idx]
            if args.dataset_names is not None
            else os.path.basename(os.path.normpath(sr_dir))
        )

        # GT presence
        has_gt = False
        gt_dir = None
        if is_placeholder_gt(raw_gt_input):
            has_gt = False
            logger.info(f"Dataset [{dataset_name}]: No GT (placeholder). NR only.")
        elif not os.path.exists(raw_gt_input):
            has_gt = False
            logger.warning(f"Dataset [{dataset_name}]: GT path not found '{raw_gt_input}'. NR only.")
        else:
            has_gt = True
            gt_dir = raw_gt_input
            logger.info(f"Dataset [{dataset_name}]: GT found at '{gt_dir}'. FR+NR.")

        # -------------------------
        # (2) detail csv path:
        # ./log/details/<dataset>/<model>_<dataset>_<timestamp>.csv
        # -------------------------
        dataset_detail_dir = os.path.join(details_root, dataset_name)
        os.makedirs(dataset_detail_dir, exist_ok=True)
        detail_csv_path = os.path.join(dataset_detail_dir, f"{model_name}_{dataset_name}_{run_ts}.csv")

        detail_header = ["Dataset", "Model", "Timestamp", "Image", "Has_GT"] + ALL_METRIC_NAMES_FIXED + ["Runtime"]
        detail_f = open(detail_csv_path, "w", newline="")
        detail_writer = csv.writer(detail_f)
        detail_writer.writerow(detail_header)

        # -------------------------
        # (3) avg csv path:
        # ./log/<dataset>.csv (append per model)
        # -------------------------
        avg_csv_path = os.path.join(root_out, f"{dataset_name}.csv")
        avg_header = ["Dataset", "Model", "Timestamp", "Num_Images", "Has_GT"] + ALL_METRIC_NAMES_FIXED

        # list images + pairing
        sr_list = list_images(sr_dir)
        if not sr_list:
            logger.warning(f"Dataset [{dataset_name}]: No images found in '{sr_dir}'.")
            detail_f.close()
            # still append avg row with empty metrics
            append_avg_csv_row(
                avg_csv_path,
                avg_header,
                [dataset_name, model_name, run_ts, 0, has_gt] + [""] * len(ALL_METRIC_NAMES_FIXED),
            )
            continue

        sr_map, sr_dup = build_stem_map(sr_list)
        if sr_dup > 0:
            logger.warning(f"Dataset [{dataset_name}]: duplicate SR stems={sr_dup}, kept first.")

        gt_map = {}
        if has_gt:
            gt_list = list_images(gt_dir)
            gt_map, gt_dup = build_stem_map(gt_list)
            if gt_dup > 0:
                logger.warning(f"Dataset [{dataset_name}]: duplicate GT stems={gt_dup}, kept first.")

        stems = sorted(sr_map.keys())

        # per-metric sum+count
        metric_sum = {m: 0.0 for m in ALL_METRIC_NAMES_FIXED}
        metric_cnt = {m: 0 for m in ALL_METRIC_NAMES_FIXED}

        processed_images = 0

        for stem in stems:
            sr_path = sr_map[stem]
            img_name = os.path.basename(sr_path)

            sr_img = cv2.imread(sr_path, cv2.IMREAD_COLOR)
            if sr_img is None:
                logger.warning(f"Failed to read SR image: {sr_path}")
                continue

            sr_tensor = img2tensor(sr_img, bgr2rgb=True, float32=True).unsqueeze(0).to(device).contiguous() / 255.0

            gt_tensor = None
            if has_gt and stem in gt_map:
                gt_img = cv2.imread(gt_map[stem], cv2.IMREAD_COLOR)
                if gt_img is not None:
                    gt_tensor = img2tensor(gt_img, bgr2rgb=True, float32=True).unsqueeze(0).to(device).contiguous() / 255.0
                else:
                    logger.warning(f"Failed to read GT: {gt_map[stem]}. FR skipped.")
            elif has_gt:
                logger.warning(f"Missing GT for SR '{img_name}' (stem='{stem}'). FR skipped.")

            current_metrics = {m: None for m in ALL_METRIC_NAMES_FIXED}

            maybe_cuda_sync(device)
            t0 = time.time()

            with torch.no_grad():
                # NR
                for name in nr_metric_names:
                    if name not in metrics_models:
                        continue
                    try:
                        val = metrics_models[name](sr_tensor).item()
                        current_metrics[name] = val
                        metric_sum[name] += val
                        metric_cnt[name] += 1
                    except Exception as e:
                        logger.error(f"Error computing {name} on {dataset_name}/{img_name}: {e}")

                # FR
                if has_gt and gt_tensor is not None:
                    for name in fr_metric_names:
                        if name not in metrics_models:
                            continue
                        try:
                            val = metrics_models[name](sr_tensor, gt_tensor).item()
                            current_metrics[name] = val
                            metric_sum[name] += val
                            metric_cnt[name] += 1
                        except Exception as e:
                            logger.error(f"Error computing {name} on {dataset_name}/{img_name}: {e}")

            maybe_cuda_sync(device)
            t1 = time.time()
            runtime = t1 - t0

            processed_images += 1

            # log line
            log_parts = [f"{dataset_name}/{img_name}"]
            for n in fr_metric_names:
                if current_metrics.get(n) is not None:
                    log_parts.append(f"{n}: {current_metrics[n]:.4f}")
            for n in nr_metric_names:
                if current_metrics.get(n) is not None:
                    log_parts.append(f"{n}: {current_metrics[n]:.4f}")
            logger.info(" | ".join(log_parts) + f" | Time: {runtime:.2f}s")

            # write detail row
            row = [dataset_name, model_name, run_ts, img_name, has_gt]
            for n in ALL_METRIC_NAMES_FIXED:
                v = current_metrics.get(n, None)
                row.append("" if v is None else v)
            row.append(f"{runtime:.4f}")
            detail_writer.writerow(row)

        detail_f.flush()
        detail_f.close()

        # averages
        avg_results = {}
        for n in ALL_METRIC_NAMES_FIXED:
            avg_results[n] = (metric_sum[n] / metric_cnt[n]) if metric_cnt[n] > 0 else None

        logger.info(f"----- Average for {dataset_name} ({model_name}) -----")
        parts = []
        for n in ALL_METRIC_NAMES_FIXED:
            v = avg_results[n]
            if v is not None:
                parts.append(f"{n}: {v:.4f}")
        logger.info(" | ".join(parts) if parts else "(No valid metrics computed)")
        logger.info("----------------------------------\n")

        # append avg row to ./log/<dataset>.csv
        row = [dataset_name, model_name, run_ts, processed_images, has_gt]
        for n in ALL_METRIC_NAMES_FIXED:
            v = avg_results[n]
            row.append("" if v is None else f"{v:.6f}")
        append_avg_csv_row(avg_csv_path, avg_header, row)

        logger.info(f"[Saved] Detail CSV: {detail_csv_path}")
        logger.info(f"[Saved] Avg CSV:    {avg_csv_path}")

    logger.info("All done.")


if __name__ == "__main__":
    main()
