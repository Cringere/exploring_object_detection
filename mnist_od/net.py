import torch
import torch.nn as nn

import mnist_od.config as config

class Net(nn.Module):
	def __init__(self, fully_convolutional, use_sigmoid):
		super().__init__()
		self.fully_convolutional = fully_convolutional
		self.use_sigmoid = use_sigmoid

		self.c = config.INPUT_CHANNELS

		def conv_block(out_c, kernel_size=3, stride=1, padding=0, use_bn=True, use_relu=True):
			s = nn.Sequential(
				nn.Conv2d(
					self.c,
					out_c,
					kernel_size=kernel_size,
					stride=stride,
					padding=padding
				),
				nn.BatchNorm2d(out_c) if use_bn else nn.Identity(),
				nn.LeakyReLU(0.1) if use_relu else nn.Identity(),
			)

			self.c = out_c
			return s
		
		def pool_block(scale=2):
			return nn.MaxPool2d(kernel_size=scale, stride=scale)

		self.out_label_size = (
			config.NUM_CLASSES +
			config.BOXES_PER_CELL * (1 + 2 + 2)
		)

		out_size = (
			config.NUM_CELLS *
			config.NUM_CELLS *
			self.out_label_size 
		)

		self.net = nn.Sequential(
			conv_block(64, 7, 2, 3),
			pool_block(2),

			conv_block(192, 3, 1, 1),
			pool_block(2),

			conv_block(128, 1, 1, 0),
			conv_block(256, 3, 1, 1),
			conv_block(256, 1, 1, 0),
			conv_block(512, 3, 1, 1),
			pool_block(2),

			conv_block(512, 1, 1, 0),
			conv_block(512, 3, 1, 1),
			pool_block(2),
		)

		if fully_convolutional:
			self.end = nn.Sequential(
				conv_block(512, 1, 1, 0),
				conv_block(self.out_label_size, 1, 1, 0, use_bn=False, use_relu=False),
			)
		else:
			self.end = nn.Sequential(
				nn.Flatten(),
				nn.Linear(512 * 4 * 4, 2048),
				nn.LeakyReLU(0.1),
				nn.Linear(2048, out_size),
			)
	
	def forward(self, x):
		'''
		input: (batches, channels, size, size)
		output: (batches, cells, cells, classes + 5 * boxes)
		'''
		y = self.end(self.net(x)) # (batches, classes + 5 * boxes, cells, cells)
		
		# convert to (batches, cells, cells, classes + 5 * boxes)
		if self.fully_convolutional:
			y = torch.permute(y, (0, 2, 3, 1))
		else:
			batches = x.shape[0]
			y = y.reshape((
				batches,
				config.NUM_CELLS,
				config.NUM_CELLS,
				self.out_label_size,
			))
		
		# optionally use sigmoid over the class probabilities and the box probability
		if self.use_sigmoid:
			# classes
			y[..., 0: config.NUM_CLASSES] = y[..., 0: config.NUM_CLASSES].sigmoid()

			# box probabilities
			for i in range(config.BOXES_PER_CELL):
				idx = config.NUM_CLASSES + i * 5
				y[..., idx] = y[..., idx].sigmoid()
		
		return y
		
