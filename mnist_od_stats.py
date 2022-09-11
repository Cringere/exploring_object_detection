import torch
import torch.nn as nn
import torch.optim as optim

from custom_datasets.mnist_od_dataset import MnistOdDataset

from mnist_od.net import Net
import mnist_od.config as config

from stater import Stater

import os
from dotenv import load_dotenv

if __name__ == '__main__':
	# load `.env` file
	load_dotenv()

	# hardware
	device = 'cuda' if torch.cuda.is_available() else 'cpu'

	# load network
	net = torch.load('model_files/mnist_od.tch')
	net = net.to(device)
	
	# data and loader
	dataset = MnistOdDataset(
		root=os.getenv('MNIST_ROOT'),
		train=True,
		download=True,
		n_cells=config.NUM_CELLS,
		out_size=config.INPUT_SIZE,
		items_per_image=config.ITEMS_PER_IMAGE,
		single_object_per_cell=True,
		dataset_size=config.STATS_NUM_ITEMS
	)

	# mean average precision helper object
	stater = Stater(
		device,
		net,
		config.NUM_CLASSES,
		dataset,
		config.STATS_NUM_ITEMS,
		config.STATS_IOU_THRESHOLDS,
		os.path.join(os.getenv('OUT_DIR'), 'voc_od_stats.png'),
		os.path.join(os.getenv('OUT_DIR'), 'voc_od_map.png'),
	)
	stater.calculate()
