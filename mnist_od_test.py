import torch

from custom_datasets.mnist_od_dataset import MnistOdDataset

from torch.utils.data import DataLoader

import numpy as np

from mnist_od.net import Net
import mnist_od.config as config

import os
from dotenv import load_dotenv

from tester import Tester

if __name__ == '__main__':
	# load `.env` file
	load_dotenv()

	# hardware
	device = 'cuda' if torch.cuda.is_available() else 'cpu'

	# load model
	net = torch.load('model_files/mnist_od.tch')

	# data
	dataset = MnistOdDataset(
		root=os.getenv('MNIST_ROOT'),
		train=False,
		download=True,
		n_cells=config.NUM_CELLS,
		out_size=config.INPUT_SIZE,
		items_per_image=config.ITEMS_PER_IMAGE,
		dataset_size=config.TEST_NUM_ITEMS,
	)

	with torch.no_grad():
		tester = Tester(
			net,
			dataset,
			config.NUM_CLASSES,
			config.NUM_CELLS,
			config.INPUT_SIZE,
			config.TEST_NUM_ITEMS,
			config.TEST_CONFIDENCE_THRESHOLD,
			os.path.join(os.getenv('OUT_DIR'), 'mnist_od_test_sample.png'),
			lambda x: x,
			device,
			thickness=1,
		)
		tester.test()