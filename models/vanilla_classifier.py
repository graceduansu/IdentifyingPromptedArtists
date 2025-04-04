import copy

import clip
import safetensors.torch
import torch.nn as nn

from models.model_utils import convert_weights_float, init_weights


class ProjectionHead(nn.Module):
    def __init__(self, num_classes, num_input_ftrs=512, hidden_dim=256):
        super(ProjectionHead, self).__init__()

        self.num_classes = num_classes
        self.num_input_ftrs = num_input_ftrs
        self.hidden_dim = hidden_dim

        self.fc_1 = nn.Linear(self.num_input_ftrs, self.hidden_dim)
        self.relu = nn.ReLU()
        self.fc_2 = nn.Linear(self.hidden_dim, num_classes)

        self.fc_1.apply(init_weights)
        self.fc_2.apply(init_weights)

    def forward(self, x):
        x = self.fc_1(x)
        x = self.relu(x)
        x = self.fc_2(x)
        return x


class ClipClassifier(nn.Module):
    def __init__(self, num_classes, model_arch="ViT-L/14"):
        super(ClipClassifier, self).__init__()

        clipmodel, _ = clip.load(model_arch)
        self.clip_visual = copy.deepcopy(clipmodel.visual)
        del clipmodel

        if model_arch == "ViT-B/16":
            feat_dim = 512
            hidden_dim = 256
        elif model_arch == "ViT-L/14":
            feat_dim = 768
            hidden_dim = 512

        self.classifier_head = ProjectionHead(
            num_classes=num_classes, num_input_ftrs=feat_dim, hidden_dim=hidden_dim
        )

        convert_weights_float(self.clip_visual)

    def forward(self, x):
        x = self.clip_visual.forward(x)
        x = self.classifier_head(x)
        return x


def load_checkpoint(ckpt_path, num_classes, model_arch="ViT-L/14"):
    model = ClipClassifier(num_classes=num_classes, model_arch=model_arch)
    st = safetensors.torch.load_file(ckpt_path)
    msg = model.load_state_dict(st)
    return model
