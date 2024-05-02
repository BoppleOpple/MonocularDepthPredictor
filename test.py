import torch
import torchvision
import numpy as np
import time

a = torchvision.models.mobilenet_v2(weights='IMAGENET1K_V2')

print(a.features)