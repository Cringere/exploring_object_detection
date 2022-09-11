from torchvision.datasets import VOCDetection
import torchvision.transforms as T

from torch.utils.data import DataLoader

from utils.drawer import Drawer

from tqdm import tqdm

from voc_od.nets import BaseNet, DetectionNet, ClassificationNet
import voc_od.config as config

from voc_od.voc_label import VocLabel

import matplotlib.pyplot as plt

import os
from dotenv import load_dotenv

def from_dataset():
	dataset = VOCDetection(
		root=os.getenv('VOC_ROOT'),
		download=os.getenv('VOC_DOWNLOAD') == 'True',
		image_set='trainval',
		transform=T.Compose([
			T.ToTensor(),
			T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
		]),
	)

	for img, label in dataset:
		drawer = Drawer.from_array_chw(img.numpy())
		for obj in label['annotation']['object']:
			box = obj['bndbox']
			xmin, xmax, ymin, ymax = box['xmin'], box['xmax'], box['ymin'], box['ymax']
			drawer.bounding_box_from_corners(
				int(xmin),
				int(ymin),
				int(xmax),
				int(ymax),
				[1, 1, 1]
			)
		
		path = os.path.join(os.getenv('OUT_DIR'), 'voc_od_data_sample.png')
		drawer.save(path)
		break

def from_loader():
	# data and loader
	def target_transform(label):
		return (VocLabel
			.from_label(label)
			.to_tensor(config.NUM_CELLS, config.CLASS_TO_INDEX)
		)

	dataset = VOCDetection(
		root=os.getenv('VOC_ROOT'),
		download=os.getenv('VOC_DOWNLOAD') == 'True',
		image_set='trainval',
		transform=T.Compose([
			T.ToTensor(),
			T.Resize((config.INPUT_SIZE, config.INPUT_SIZE)),
			T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
		]),
		target_transform=target_transform,
	)

	loader = DataLoader(
		dataset,
		1,
		True,
	)

	for img, label in loader:
		img = img[0].numpy()
		label = label[0].numpy()

		drawer = Drawer.from_array_chw(img)

		cell_size = config.INPUT_SIZE / config.NUM_CELLS

		for i in range(label.shape[0]):
			for j in range(label.shape[1]):
				box = label[i, j, config.NUM_CLASSES:]

				if box[0] < 0.5:
					continue

				center_x, center_y, width, height = box[1:]
				
				center_x = j * cell_size + center_x * cell_size
				center_y = i * cell_size + center_y * cell_size

				width *= cell_size
				height *= cell_size
				
				print(center_x, center_y, width, height)
				drawer.bounding_box_from_corners(
					int(center_x - width / 2),
					int(center_y - height / 2),
					int(center_x + width / 2),
					int(center_y + height / 2),
					[1, 1, 1]
				)
		
		path = os.path.join(os.getenv('OUT_DIR'), 'voc_od_data_sample.png')
		drawer.save(path)
		break

if __name__ == '__main__':
	load_dotenv()
	# from_dataset()
	from_loader()
	