import torch

import voc_od.config as config

from voc_od.nets import *

from utils import count_parameters

if __name__ == '__main__':
	base_net = BaseNet()
	classification_net = ClassificationNet(base_net, 10)
	detection_net = DetectionNet(base_net)

	x = torch.zeros((2, config.INPUT_CHANNELS, config.INPUT_SIZE, config.INPUT_SIZE))
	print(f'input shape: {x.shape}')
	print()

	print('base net')
	count_parameters(base_net)
	print(f'output shape: {base_net(x).shape}')
	print()

	print('classification net')
	count_parameters(classification_net)
	print(f'output shape: {classification_net(x).shape}')
	print()

	print('detection net')
	count_parameters(detection_net)
	print(f'output shape: {detection_net(x).shape}')
	print()
