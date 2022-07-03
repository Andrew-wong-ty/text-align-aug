# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
from collections import OrderedDict

import torch.nn.functional as F
from functools import partial
from transformers import ViTFeatureExtractor, ViTModel

def build_backbone_vit(config):
    model = ViTModel.from_pretrained(config.backbone)
    return model
