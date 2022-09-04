import torch

import mnist_od.config as config

from mnist_od.net import Net
from custom_datasets.mnist_od_dataset import MnistOdDataset

from torch.utils.data import DataLoader

def test_net(net):
	learnable_parameters = sum(p.numel() for p in net.parameters() if p.requires_grad)
	n_params = "{:,}".format(learnable_parameters)
	print(f'model has {n_params} learnable parameters')

	x = torch.randn((1, config.INPUT_CHANNELS, config.INPUT_SIZE, config.INPUT_SIZE))
	y = net(x)
	print(f'input:  {x.shape}')
	print(f'output: {y.shape}')


if __name__ == '__main__':
	net = Net(fully_convolutional=True, use_sigmoid=True)
	test_net(net)
	print()
	net = Net(fully_convolutional=False, use_sigmoid=True)
	test_net(net)
