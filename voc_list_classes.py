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

dataset = torchvision.datasets.VOCDetection(
	root=os.getenv('VOC_ROOT'),
	image_set='val',
	download=os.getenv('VOC_DOWNLOAD') == 'True',
	target_transform=lambda label: VocLabel.from_label(label),
)

names = set()
for _, label in tqdm(dataset):
	for obj in label.label_objects:
		names.add(obj.name)

for i, name in enumerate(names):
	print(i, name)

# output:
# 0 car
# 1 cow
# 2 tvmonitor
# 3 train
# 4 horse
# 5 sofa
# 6 chair
# 7 dog
# 8 bottle
# 9 person
# 10 motorbike
# 11 boat
# 12 bird
# 13 sheep
# 14 cat
# 15 aeroplane
# 16 bus
# 17 diningtable
# 18 pottedplant
# 19 bicycle