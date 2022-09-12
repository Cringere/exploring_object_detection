from typing import Literal, Union
import torch

import torchvision.transforms as T
from torch.utils.data import DataLoader

import matplotlib.pyplot as plt

from utils import iou_c_stats, mean_average_precision, iou_c_stats_absolute

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
			num_repeats,
			iou_thresholds,
			stats_file_path,
			map_file_path,
			stats_mode: Union[Literal['relative'], Literal['absolute']]
			):
		assert stats_mode == 'relative' or stats_mode == 'absolute'

		self.device = device
		self.net = net
		self.num_classes = num_classes
		self.dataset = dataset
		self.iou_thresholds = iou_thresholds
		self.num_test_items = num_test_items
		self.num_repeats = num_repeats
		self.stats_file_path = stats_file_path
		self.map_file_path = map_file_path
		self.stats_mode = stats_mode
		
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
			self.net.eval()

			all_labels = None
			all_preds = None

			for i, (img, label) in enumerate(self.dataloader):
				img = img.to(self.device)
				pred = self.net(img).cpu()
				
				if all_labels is None:
					all_labels = label
				else:
					all_labels = torch.cat((all_labels, label), dim=0)
				
				if all_preds is None:
					all_preds = pred
				else:
					all_preds = torch.cat((all_preds, pred), dim=0)
				
				if i > self.num_repeats:
					break
			
			pred = all_preds
			label = all_labels
			
			if self.stats_mode == 'relative':
				return iou_c_stats(
					label,
					pred,
					iou_thresholds=self.iou_thresholds
				)
			else:
				stats = iou_c_stats_absolute.Stats(
					pred.shape,
					label.shape,
					iou_thresholds=self.iou_thresholds
				) 

				return stats.iou_c_stats(pred, label)
