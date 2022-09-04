import torch

from custom_datasets.mnist_od_dataset import MnistOdDataset

from torch.utils.data import DataLoader

import numpy as np

from mnist_od.net import Net
import mnist_od.config as config

import os
from dotenv import load_dotenv

from utils.drawer import Drawer

class Tester:
	def __init__(self):
		# hardware
		self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

		# load model
		self.net = torch.load('model_files/mnist_od.tch').to(self.device)

		# dataset on test mode
		self.dataset = MnistOdDataset(
			root=os.getenv('MNIST_ROOT'),
			train=False,
			download=True,
			n_cells=config.NUM_CELLS,
			out_size=config.INPUT_SIZE,
			items_per_image=config.ITEMS_PER_IMAGE,
			dataset_size=config.TEST_NUM_ITEMS,
		)

		# dataloader with the same size as the dataset
		self.dataloader = DataLoader(
			dataset=self.dataset,
			batch_size=config.TEST_NUM_ITEMS,
			shuffle=True,
		)

	def test(self):
		drawers = []
		for imgs, labels in self.dataloader:
			preds = self.net(imgs.to(self.device)).cpu()
			for item in range(imgs.shape[0]):
				img = imgs[item].numpy()
				label = labels[item].numpy()
				pred = preds[item].numpy()

				drawer = self.test_item(img, label, pred)
				drawer.border((1, 1, 1))
				drawers.append(drawer)
			break
				
		out_image = Drawer.concat_to_grid(drawers, columns=4)

		from skimage.io import imsave
		path = os.path.join(os.getenv('OUT_DIR'), 'mnist_od_test_sample.png')
		imsave(path, out_image)

	def test_item(self, img, label, pred):
		drawer = Drawer.from_array_chw(img)

		# draw real bounding boxes
		for i in range(label.shape[0]):
			for j in range(label.shape[1]):
				l = label[i, j] # (classes, 5)
				self.draw_bounding_box(i, j, drawer, l, (1, 0.1, 0.2))
		
		# draw predicted bounding boxes
		for i in range(pred.shape[0]):
			for j in range(pred.shape[1]):
				l = pred[i, j].tolist() # (classes + 5 * b)
				classes = l[:config.NUM_CLASSES]
				boxes = l[config.NUM_CLASSES:]
				for b in range(len(boxes) // 5):
					box_l = classes + boxes[b * 5: (b + 1) * 5]
					p = max(min(boxes[b * 5], 1.0), 0.0)
					color = (0.2, 1.0, 0.2)
					# color = list(map(lambda x: x * (p - threshold) / (1.0 - threshold), color))
					color = list(map(lambda x: x * p, color))
					self.draw_bounding_box(i, j, drawer, box_l, color)

		return drawer

	def draw_bounding_box(self, i, j, drawer, label, color):
		p = label[10]
		center = label[11: 11 + 2]
		size = label[13: 13 + 2]
		if p > config.TEST_CONFIDENCE_THRESHOLD:
			tl_x, tl_y, br_x, br_y = self.dataset.relative_box_to_absolute(
				i,
				j,
				center[0],
				center[1],
				size[0],
				size[1]
			)
			
			drawer.bounding_box_from_corners(tl_x, tl_y, br_x, br_y, color)

if __name__ == '__main__':
	# load `.env` file
	load_dotenv()

	with torch.no_grad():
		tester = Tester()
		tester.test()
