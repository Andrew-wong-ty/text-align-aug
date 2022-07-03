import torch
from typing import Optional, List
from torch import Tensor

import json
import os


def read_json(file_name):
    with open(file_name) as handle:
        out = json.load(handle)
    return out


