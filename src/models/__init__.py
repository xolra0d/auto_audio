import json
from pathlib import Path

import omegaconf
import torch

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

from .modeling_gigaam import GigaAMASR, GigaAMEmo


def load_stt(
    config_path="models/stt/v3_rnnt.json",
    checkpoint_path="models/stt/pytorch_model.bin",
    device: str | None = None,
) -> GigaAMASR:
    with open(config_path) as f:
        config = json.load(f)

    model_cfg = omegaconf.OmegaConf.create(config)
    assert model_cfg is not omegaconf.ListConfig
    model = GigaAMASR(cfg=model_cfg)
    checkpoint = torch.load(checkpoint_path)
    new_checkpoint = {}
    for key, value in checkpoint.items():
        if key.startswith("model."):
            new_key = key[6:]  # Remove "model." prefix (6 characters)
            new_checkpoint[new_key] = value
        else:
            new_checkpoint[key] = value
    model.load_state_dict(new_checkpoint)
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # type: ignore
    model = model.to(device)
    model.eval()
    return model


def load_ste(
    config_path="models/ste/emo.json",
    checkpoint_path="models/ste/emo.ckpt",
    device: str | None = None,
) -> GigaAMEmo:
    with open(config_path) as f:
        config = json.load(f)

    model_cfg = omegaconf.OmegaConf.create(config)
    assert model_cfg is not omegaconf.ListConfig
    model = GigaAMEmo(model_cfg)
    checkpoint = torch.load(checkpoint_path, weights_only=False)
    model.load_state_dict(checkpoint["state_dict"])
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()
    return model
