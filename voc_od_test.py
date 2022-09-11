import torch

from torchvision.datasets import VOCDetection

import torchvision.transforms as T

from voc_od.nets import DetectionNet
from voc_od.voc_label import VocLabel
import voc_od.config as config

import os
from dotenv import load_dotenv

from tester import Tester

if __name__ == '__main__':
	# load `.env` file
	load_dotenv()

	# hardware
	device = 'cuda' if torch.cuda.is_available() else 'cpu'

	# load model
	net = torch.load(os.path.join('model_files/', config.DETECTION_MODEL_NAME))

	# dataset
	def target_transform(label):
		return (VocLabel
			.from_label(label)
			.to_tensor(config.NUM_CELLS, config.CLASS_TO_INDEX)
		)

	dataset = VOCDetection(
		root=os.getenv('VOC_ROOT'),
		download=os.getenv('VOC_DOWNLOAD') == 'True',
		image_set='val',
		transform=T.Compose([
			T.ToTensor(),
			T.Resize((config.INPUT_SIZE, config.INPUT_SIZE)),
			T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
		]),
		target_transform=target_transform,
	)

	# run the tester
	with torch.no_grad():
		tester = Tester(
			net,
			dataset,
			config.NUM_CLASSES,
			config.NUM_CELLS,
			config.INPUT_SIZE,
			config.TEST_NUM_ITEMS,
			config.TEST_CONFIDENCE_THRESHOLD,
			os.path.join(os.getenv('OUT_DIR'), 'voc_od_test_sample.png'),
			lambda img: img * 0.5 + 0.5,
			device,
			thickness=2,
		)
		tester.test()
