import copy

import clip
import numpy as np
import safetensors.torch
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.model_utils import convert_weights_float


class ProtoClip(nn.Module):
    def __init__(self, prototype, model_arch="ViT-L/14", temperature=0.07):
        super(ProtoClip, self).__init__()

        print(f"ProtoClip init: Loading CLIP model {model_arch}")
        clipmodel, _ = clip.load(model_arch)
        self.encoder = copy.deepcopy(clipmodel.visual)
        self.encoder.proj = None
        del clipmodel

        if model_arch == "ViT-L/14":
            feat_dim = 1024
        elif model_arch == "ViT-B/16":
            feat_dim = 768
        else:
            raise NotImplementedError(
                f"Model architecture {model_arch} not implemented")

        # normalize the prototype
        assert len(prototype.shape) == 2 and prototype.shape[1] == feat_dim, \
            f'prototype shape should be (num_class, feat_dim) but got {prototype.shape}'
        with torch.no_grad():
            prototype = F.normalize(prototype, dim=1, p=2)
        # shape (num_class, feat_dim)
        self.register_buffer('prototype', prototype)
        self.temperature = temperature

        convert_weights_float(self.encoder)

    def forward(self, x):
        feat = self.encoder(x)
        feat = F.normalize(feat, dim=1, p=2)
        logits = feat @ self.prototype.T / self.temperature
        return logits  # B, num_class


def load_checkpoint(ckpt_path, prototype_path, model_arch="ViT-L/14", device="cuda"):
    prototype = torch.from_numpy(np.load(prototype_path)).to(device)
    model = ProtoClip(prototype=prototype, model_arch=model_arch)
    st = safetensors.torch.load_file(ckpt_path)
    # remove training prototype from state_dict
    st = {k: v for k, v in st.items() if "prototype" not in k}
    msg = model.load_state_dict(st, strict=False)
    print(f"Loaded model with message: {msg}")
    print(f"Model prototype shape: {model.prototype.shape}")
    print(f"Model prototype path: {prototype_path}")
    return model
