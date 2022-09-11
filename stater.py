
import torch

import torchvision.transforms as T
from torch.utils.data import DataLoader

import matplotlib.pyplot as plt

from utils import iou_c_stats_absolute, mean_average_precision

class Stater:
	'''
	Helper class for calculating the mean average precision along with the statistics
	leading to it.
	'''
	def __init__(
			self,
			device,
			net,
			num_classes,
			dataset,
			num_test_items,
			iou_thresholds,
			stats_file_path,
			map_file_path,
			):
		self.device = device
		self.net = net
		self.num_classes = num_classes
		self.dataset = dataset
		self.iou_thresholds = iou_thresholds
		self.num_test_items = num_test_items
		self.stats_file_path = stats_file_path
		self.map_file_path = map_file_path
		
		self.net = net.to(device)

		self.dataloader = DataLoader(
			dataset=self.dataset,
			batch_size=num_test_items,
			shuffle=True,
		)

	def calculate(self):
		iou_c_stats = self.generate_stats()

		nrows = self.num_classes
		ncols = len(self.iou_thresholds)

		fig, ax = plt.subplots(
			nrows = nrows,
			ncols = ncols,
			figsize=(30, 40),
		)
		fig.suptitle(f'precision, accuracy, threshold vs recall')

		for i, c in enumerate(range(self.num_classes)):
			for j, thresh in enumerate(self.iou_thresholds):
				stats = iou_c_stats[thresh][c]

				ax[i][j].plot(stats['r'], stats['p'], label='precision')
				ax[i][j].plot(stats['r'], stats['t'], label='threshold')

				ax[i][j].set_title(f'iou thresh: {thresh}, class: {c}')
				ax[i][j].set_xlabel('recall')
				ax[i][j].legend()

		plt.tight_layout()
		plt.savefig(self.stats_file_path)

		ious, m_ap = mean_average_precision(iou_c_stats)
		plt.clf()
		plt.figure(figsize=(5, 5))
		plt.figure()
		plt.plot(ious, m_ap)
		plt.title('mean average precision')
		plt.savefig(self.map_file_path)

	def generate_stats(self):
		with torch.no_grad():
			for img, label in self.dataloader:
				img = img.to(self.device)
				label = label.to(self.device)
				pred = self.net(img)
				return iou_c_stats_absolute(
					label.cpu(),
					pred.cpu(),
					iou_thresholds=self.iou_thresholds
				)

