import torch
import torch.nn as nn
import torch.optim as optim

from custom_datasets.mnist_od_dataset import MnistOdDataset

from custom_losses.yolo_loss import YoloLoss

from torch.utils.data import DataLoader

from tqdm import tqdm

from mnist_od.net import Net
import mnist_od.config as config

import matplotlib.pyplot as plt

import os
from dotenv import load_dotenv

class Trainer:
	def __init__(self):
		# hardware
		self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

		# network
		if os.getenv('MNIST_LOAD_MODEL') == 'True':
			self.net = torch.load('model_files/mnist_od.tch')
		else:
			self.net = Net(config.FULLY_CONVOLUTIONAL, config.USE_SIGMOID)
		self.net = self.net.to(self.device)

		# data and loader
		self.dataset = MnistOdDataset(
			root=os.getenv('MNIST_ROOT'),
			train=True,
			download=os.getenv('MNIST_DOWNLOAD') == 'True',
			n_cells=config.NUM_CELLS,
			out_size=config.INPUT_SIZE,
			items_per_image=config.ITEMS_PER_IMAGE,
		)
		self.dataloader = DataLoader(
			dataset=self.dataset,
			batch_size=config.BATCH_SIZE,
			shuffle=True,
		)

		# optimizer and loss
		self.optimizer = optim.Adam(
			self.net.parameters(),
			lr=config.LEARNING_RATE
		)
		self.loss_fn = YoloLoss(config.NUM_CLASSES)

		# history
		self.location_losses = []
		self.size_losses = []
		self.prob_loss_obj = []
		self.prob_loss_no_obj = []
		self.classes_loss = []
		self.total_losses = []

	def train(self):
		for epoch in range(config.EPOCHS):
			for (img, label) in tqdm(self.dataloader, desc=f'epoch {epoch}'):
				# img: (batch, channels, w, h)
				img = img.to(self.device)
				
				# label: (batch, cells, cells, classes + 5)
				label = label.to(self.device)

				# pred: (batch, cells, cells, classes + 5 * b)
				pred = self.net(img)


				(
					location_loss,
					size_loss,
					prob_loss_obj,
					prob_loss_no_obj,
					classes_loss,
				) = self.loss_fn(label, pred)

				# combine the losses
				total_loss = (
					config.LAMBDA_COORD * location_loss +
					config.LAMBDA_COORD * size_loss +
					config.LAMBDA_OBJ * prob_loss_obj +
					config.LAMBDA_NO_OBJ * prob_loss_no_obj +
					config.LAMBDA_PROB * classes_loss
				)

				# optimizer step
				self.optimizer.zero_grad()
				total_loss.backward()
				self.optimizer.step()

				# record
				self.location_losses.append(location_loss.item())
				self.size_losses.append(size_loss.item())
				self.prob_loss_obj.append(prob_loss_obj.item())
				self.prob_loss_no_obj.append(prob_loss_no_obj.item())
				self.classes_loss.append(classes_loss.item())
				self.total_losses.append(total_loss.item())

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

def test_dims():
	pred = torch.Tensor([
		# batch 1
		[
			# row 1
			[
				# column 1
				[
					# box 1
					[1, 2, 3],
					# box 2
					[4, 5, 6],
				],
				# column 2
				[
					# box 1
					[7, 8, 9],
					# box 2
					[1, 1, 1],
				],
			],
			# row 1
			[
				# column 1
				[
					# box 1
					[-1, -2, -3],
					# box 2
					[-4, -5, -6],
				],
				# column 2
				[
					# box 1
					[-7, -8, -9],
					# box 2
					[-1, -1, -1],
				],
			],
		],
	])

	indices = torch.Tensor([
		# batch 1
		[
			# row 1
			[
				# column 1
				[
					# index
					0
				],
				# column 2
				[
					# index
					1
				],
			],
			# row 1
			[
				# column 1
				[
					# index
					1
				],
				# column 2
				[
					# index
					0
				],
			],
		],
	])

	print(pred.shape)
	print(indices.shape)

	indices = indices.unsqueeze(-1)
	indices = indices.repeat((1, 1, 1, 1, 3)).long()
	print(indices.shape)

	target = torch.gather(pred, 3, indices)
	print(target.shape)
	target = target.squeeze(3)
	print(target.shape)
	print(target)

if __name__ == '__main__':
	# load `.env` file
	load_dotenv()

	# tests
	# test_dims()

	# train
	trainer = Trainer()
	trainer.train()

	# plot losses
	trainer.plot_losses()
	offset=60
	trainer.plot_losses(offset, name=f'mnist_od_losses_offset_{offset}.png')

	# save model
	if os.getenv('MNIST_SAVE_MODEL') == 'True':
		torch.save(trainer.net, 'model_files/mnist_od.tch')
