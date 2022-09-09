import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as T
from torch.utils.data import DataLoader

from tqdm import tqdm

from voc_od.voc_label import VocLabel

import os

from dotenv import load_dotenv
load_dotenv()

dataset = torchvision.datasets.ImageNet(
	root=os.getenv('IMAGE_NET_ROOT'),
	split='val',
)

c = [0, 0]
for _, label in tqdm(dataset):
	c[0] = min(c[0], label)
	c[1] = max(c[1], label)

print(c)

# output:
# [0, 999]
