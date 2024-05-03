import torch
import torchvision
import numpy as np
import time

a = torchvision.datasets.Kitti('data/', train=True, download=True)

print(a[0])