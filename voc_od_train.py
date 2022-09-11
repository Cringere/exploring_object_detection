import torch
import torch.optim as optim
import torch.nn as nn
from torchvision.datasets import VOCDetection, ImageNet
import torchvision.transforms as T

from custom_losses.yolo_loss import YoloLoss

from torch.utils.data import DataLoader

from tqdm import tqdm

from voc_od.nets import BaseNet, DetectionNet, ClassificationNet
import voc_od.config as config

from voc_od.voc_label import VocLabel

import matplotlib.pyplot as plt

import os
from dotenv import load_dotenv

class Trainer:
	def __init__(self):
		# hardware
		self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

		# classification history
		self.classification_losses = []

		# detection history
		self.location_losses = []
		self.size_losses = []
		self.prob_loss_obj = []
		self.prob_loss_no_obj = []
		self.classes_loss = []
		self.total_losses = []

		# load env variables
		load_stage = os.getenv('VOC_LOAD_STAGE')
		save_models = os.getenv('VOC_SAVE_MODELS').replace(' ', '').split(',')
		num_workers = os.getenv('NUM_WORKERS')
		self.num_workers = int(num_workers) if num_workers is not None else 0

		if load_stage == 'None':
			# train classification
			self.net = ClassificationNet(None, config.NUM_CLASSIFICATION_CLASSES).to(self.device)
			self.net = self.net.to(self.device)

			self.train_classification()
			self.plot_classification_losses()
			self.plot_classification_losses(100)

			# save classifier's base
			if 'Base' in save_models:
				torch.save(self.net.base, f'model_files/{config.BASE_MODEL_NAME}')
		if load_stage == 'Base':
			base = torch.load(f'model_files/{config.BASE_MODEL_NAME}').to(self.device)
			self.net = DetectionNet(base)
		if load_stage == 'Full':
			self.net = torch.load(f'model_files/{config.DETECTION_MODEL_NAME}').to(self.device)
		
		# train detection
		self.net = DetectionNet(self.net.base)
		self.net = self.net.to(self.device)
		self.train_detection()
		self.plot_detection_losses()
		self.plot_detection_losses(100)

		# save the full detection model
		if 'Full' in save_models:
			torch.save(self.net, f'model_files/{config.DETECTION_MODEL_NAME}')

	def train_classification(self):
		# data and loader
		dataset = ImageNet(
			os.getenv('IMAGE_NET_ROOT'),
			split='train',
			transform=T.Compose([
				T.ToTensor(),
				T.Resize((config.INPUT_SIZE, config.INPUT_SIZE)),
				T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
			]),
		)

		dataloader = DataLoader(
			dataset=dataset,
			batch_size=config.C_BATCH_SIZE,
			shuffle=True,
			num_workers=self.num_workers,
		)

		# optimizer and loss
		optimizer = optim.Adam(
			self.net.parameters(),
			lr=config.C_LEARNING_RATE,
		)
		loss_fn = nn.CrossEntropyLoss()

		# training
		for epoch in range(config.C_EPOCHS):
			for img, label in tqdm(dataloader, desc=f'classification epoch {epoch}'):
				img = img.to(self.device)
				label = label.to(self.device)

				# prediction and loss
				pred = self.net(img)
				loss = loss_fn(pred, label)
				
				# optimizer step
				optimizer.zero_grad()
				loss.backward()
				optimizer.step()

				# record
				self.classification_losses.append(loss.item())

	def train_detection(self):
		# data and loader
		def target_transform(label):
			return (VocLabel
				.from_label(label)
				.to_tensor(config.NUM_CELLS, config.CLASS_TO_INDEX)
			)

		dataset = VOCDetection(
			root=os.getenv('VOC_ROOT'),
			download=os.getenv('VOC_DOWNLOAD') == 'True',
			image_set=config.VOC_IMAGE_SET,
			transform=T.Compose([
				T.ToTensor(),
				T.Resize((config.INPUT_SIZE, config.INPUT_SIZE)),
				T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
			]),
			target_transform=target_transform,
		)

		dataloader = DataLoader(
			dataset=dataset,
			batch_size=config.D_BATCH_SIZE,
			shuffle=True,
			num_workers=self.num_workers,
		)

		# optimizer and loss
		optimizer = optim.Adam(
			self.net.parameters(),
			lr=config.D_LEARNING_RATE
		)
		loss_fn = YoloLoss(config.NUM_CLASSES, self.device)

		# train
		for epoch in range(config.D_EPOCHS):
			for (img, label) in tqdm(dataloader, desc=f'detection epoch {epoch}'):
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
				) = loss_fn(label, pred)

				# combine the losses
				total_loss = (
					config.LAMBDA_COORD * location_loss +
					config.LAMBDA_COORD * size_loss +
					config.LAMBDA_OBJ * prob_loss_obj +
					config.LAMBDA_NO_OBJ * prob_loss_no_obj +
					config.LAMBDA_PROB * classes_loss
				)

				# optimizer step
				optimizer.zero_grad()
				total_loss.backward()
				optimizer.step()

				# record
				self.location_losses.append(location_loss.item())
				self.size_losses.append(size_loss.item())
				self.prob_loss_obj.append(prob_loss_obj.item())
				self.prob_loss_no_obj.append(prob_loss_no_obj.item())
				self.classes_loss.append(classes_loss.item())
				self.total_losses.append(total_loss.item())

	def plot_detection_losses(self, offset=0):
		'''
		Plots the losses and saves them into `$OUT_DIR/`.
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
		plt.title('detection losses')
		plt.legend()
		name = f'voc_od_detection_losses_{offset}.png'
		plt.savefig(os.path.join(os.getenv('OUT_DIR'), name))
	
	def plot_classification_losses(self, offset=0):
		'''
		Plots the losses and saves them into `$OUT_DIR/<name>`.
		The loss can oscillate significantly at the beginning of learning, so it
		can be useful to plot the losses only after some iterations.
		'''
		plt.clf()
		plt.plot(self.classification_losses[offset:])
		plt.title('classification loss')
		name = f'voc_od_classification_losses_{offset}.png'
		plt.savefig(os.path.join(os.getenv('OUT_DIR'), name))

if __name__ == '__main__':
	load_dotenv()
	trainer = Trainer()
