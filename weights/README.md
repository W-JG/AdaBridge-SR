# AdaBridge-SR Weights

Download `final.tar.gz` from [GitHub Releases](https://github.com/W-JG/AdaBridge-SR/releases).

| Archive | Location | SHA256 |
| --- | --- | --- |
| `final.tar.gz` | `weights/final/` | `c0d1a88e169c29b508dcf7f66551fe5bae6172a55473abf8f90661b41eee74bd` |

Check the download and extract it from the repository root:

```bash
wget https://github.com/W-JG/AdaBridge-SR/releases/latest/download/final.tar.gz
sha256sum final.tar.gz
tar -xzf final.tar.gz -C weights
```

`final/` contains the U-Net and VAE encoder LoRA adapters and `trajectory_control_predictor.pt`. The same checkpoint supports one-step and two-step inference.

Download `stable-diffusion-2-1-base` separately and set `SD_PATH`. DINOv3 weights are required only for training. Neither backbone is included in the archive. The adapters are trained on Stable Diffusion 2.1-base and are subject to the use restrictions of the CreativeML Open RAIL++-M License.
