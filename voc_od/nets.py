from typing import Union
import torch
import torch.nn as nn

import voc_od.config as config

class ResConvBlock(nn.Module):
	def __init__(self, in_c, out_c, kernel_size, stride, padding):
		super().__init__()
		self.seq = nn.Sequential(
			nn.Conv2d(in_c, out_c, kernel_size, stride, padding),
			nn.BatchNorm2d(out_c),
			nn.ReLU(),
			nn.Conv2d(out_c, out_c, 1, 1, 0)
		)
		if in_c == out_c:
			self.match = nn.Identity()
		else:
			self.match = nn.Conv2d(in_c, out_c, kernel_size, stride, padding)
		
		self.leaky = nn.LeakyReLU(0.1)
	
	def forward(self, x):
		return self.leaky(self.seq(x) + self.match(x))

class ConvBLock(nn.Module):
	def __init__(self, in_c, out_c, kernel_size, stride, padding, use_bn=True, use_relu=True):
		super().__init__()
		self.seq = nn.Sequential(
			nn.Conv2d(in_c, out_c, kernel_size, stride, padding),
			nn.BatchNorm2d(out_c) if use_bn else nn.Identity(),
			nn.ReLU() if use_relu else nn.Identity(),
		)
	
	def forward(self, x):
		return self.seq(x)

class BaseNet(nn.Module):
	def __init__(self):
		super().__init__()
		self.c = config.INPUT_CHANNELS

		def conv_block(out_c, kernel_size, stride, padding):
			s = ConvBLock(self.c, out_c, kernel_size, stride, padding)
			self.c = out_c
			return s

		def res_conv_block(out_c, kernel_size, stride, padding):
			s = ResConvBlock(self.c, out_c, kernel_size, stride, padding)
			self.c = out_c
			return s
		
		def pool_block(scale=2):
			return nn.MaxPool2d(kernel_size=scale, stride=scale)
	
		self.seq = nn.Sequential(
			conv_block(64, 7, 2, 3),
			pool_block(2),

			res_conv_block(192, 3, 1, 1),
			pool_block(2),

			res_conv_block(128, 1, 1, 0),
			res_conv_block(256, 3, 1, 1),
			res_conv_block(256, 1, 1, 0),
			res_conv_block(512, 3, 1, 1),
			pool_block(2),

			res_conv_block(512, 1, 1, 0),
			res_conv_block(512, 3, 1, 1),
			pool_block(2),
		)

	def forward(self, x):
		return self.seq(x)

class ClassificationNet(nn.Module):
	def __init__(self, base: Union[BaseNet, None], n_classification_classes: int):
		super().__init__()
		if base is None:
			self.base = BaseNet()
		else:
			self.base = base
		
		self.head = nn.Sequential(
			nn.Conv2d(512, 8, 1, 1, 0),
			nn.ReLU(),
			nn.Flatten(),
			nn.Linear(8 * 8 * 8, n_classification_classes),
		)
	
	def forward(self, x):
		return self.head(self.base(x))

class DetectionNet(nn.Module):
	def __init__(self, base: Union[BaseNet, None]):
		super().__init__()
		if base is None:
			self.base = BaseNet()
		else:
			self.base = base
		
		self.out_label_size = (
			config.NUM_CLASSES +
			config.BOXES_PER_CELL * (1 + 2 + 2)
		)

		self.head = nn.Sequential(
			ResConvBlock(512, 512, 1, 1, 0),
			nn.LeakyReLU(0.1),
			ConvBLock(512, self.out_label_size, 1, 1, 0, False, False),
		)
	
	def forward(self, x):
		y = self.head(self.base(x))

		# convert to (batches, cells, cells, classes + 5 * boxes)
		y = torch.permute(y, (0, 2, 3, 1))

		# class probabilities
		y[..., 0: config.NUM_CLASSES] = y[..., 0: config.NUM_CLASSES].sigmoid()

		# box probabilities
		for i in range(config.BOXES_PER_CELL):
			idx = config.NUM_CLASSES + i * 5
			y[..., idx] = y[..., idx].sigmoid()
		
		return y
