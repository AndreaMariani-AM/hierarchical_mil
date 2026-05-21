
from __future__ import annotations

from contextlib import nullcontext
from typing import Callable

import numpy as np
import torch
from torch.utils.data import DataLoader
from wsidata import WSIData
from lazyslide.models import ImageModel
from torchvision.transforms.v2 import (
            Compose,
            Normalize,
            Resize,
            ToDtype,
            ToImage,
        )


def get_transform():
    """
    ImageNet transformation for feature extraction
    """
    transform = Compose(
        [
            ToImage(),
            ToDtype(dtype=torch.float32, scale=True),
            Resize(size=(224, 224), antialias=False),
            Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ]
            )
    
    return transform
