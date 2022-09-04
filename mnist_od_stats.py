import torch
import torch.nn as nn
import torch.optim as optim

from custom_datasets.mnist_od_dataset import MnistOdDataset

from torch.utils.data import DataLoader

from mnist_od.net import Net
import mnist_od.config as config

import matplotlib.pyplot as plt

import os
from dotenv import load_dotenv

from utils import iou_c_stats, mean_average_precision

class Stater:
	def __init__(self, iou_thresholds):
		self.iou_thresholds = iou_thresholds

		# hardware
		self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

		# load network
		self.net = torch.load('model_files/mnist_od.tch')
		self.net = self.net.to(self.device)

		# data and loader
		self.dataset = MnistOdDataset(
			root=os.getenv('MNIST_ROOT'),
			train=True,
			download=True,
			n_cells=config.NUM_CELLS,
			out_size=config.INPUT_SIZE,
			items_per_image=config.ITEMS_PER_IMAGE,
			single_object_per_cell=True,
			dataset_size=config.STATS_NUM_ITEMS
		)
		self.dataloader = DataLoader(
			dataset=self.dataset,
			batch_size=config.STATS_NUM_ITEMS,
			shuffle=True,
		)

	def test(self):
		iou_c_stats = self.generate_stats()

		nrows = config.NUM_CLASSES
		ncols = len(self.iou_thresholds)

		fig, ax = plt.subplots(
			nrows = nrows,
			ncols = ncols,
			figsize=(30, 40),
		)
		fig.suptitle(f'precision, accuracy, threshold vs recall')

		for i, c in enumerate(range(config.NUM_CLASSES)):
			for j, thresh in enumerate(self.iou_thresholds):
				stats = iou_c_stats[thresh][c]

				ax[i][j].plot(stats['r'], stats['p'], label='precision')
				ax[i][j].plot(stats['r'], stats['t'], label='threshold')

				ax[i][j].set_title(f'iou thresh: {thresh}, class: {c}')
				ax[i][j].set_xlabel('recall')
				ax[i][j].legend()

		name = 'mnist_do_stats.png'
		plt.tight_layout()
		plt.savefig(os.path.join(os.getenv('OUT_DIR'), name))

		ious, m_ap = mean_average_precision(iou_c_stats)
		plt.clf()
		plt.figure(figsize=(5, 5))
		plt.figure()
		plt.plot(ious, m_ap)
		plt.title('mean average precision')
		name = 'mnist_od_mean_average_precision.png'
		plt.savefig(os.path.join(os.getenv('OUT_DIR'), name))

	def generate_stats(self):
		with torch.no_grad():
			for img, label in self.dataloader:
				img = img.to(self.device)
				label = label.to(self.device)
				pred = self.net(img)
				return iou_c_stats(
					config.NUM_CLASSES,
					label.cpu(),
					pred.cpu(),
					iou_thresholds=self.iou_thresholds
				)

	def plot_losses(self, offset=0, name='mnist_od_losses.png'):
		'''
		Plots the losses and saves them into `$OUT_DIR/<name>`.
		The loss can oscillate significantly at the beginning of learning, so it
		can be useful to plot the losses only after some iterations.
		'''
		plt.clf()
		plt.plot(self.location_losses[offset:], label='location_losses')
		plt.plot(self.size_losses[offset:], label='size_losses')
		plt.plot(self.prob_loss_obj[offset:], label='prob_loss_obj')
		plt.plot(self.prob_loss_no_obj[offset:], label='prob_loss_no_obj')
		plt.plot(self.classes_loss[offset:], label='classes_loss')
		plt.plot(self.total_losses[offset:], label='total_losses')
		plt.title('losses')
		plt.legend()
		plt.savefig(os.path.join(os.getenv('OUT_DIR'), name))

if __name__ == '__main__':
	# load `.env` file
	load_dotenv()

	# mean average precision helper object
	stater = Stater(config.STATS_IOU_THRESHOLDS)
	stater.test()
