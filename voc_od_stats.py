import torch

from torchvision.datasets import VOCDetection
import torchvision.transforms as T

from voc_od.nets import DetectionNet
import voc_od.config as config
from voc_od.voc_label import VocLabel

from stater import Stater


import os
from dotenv import load_dotenv

from utils import iou_c_stats_absolute, mean_average_precision

if __name__ == '__main__':
	# load `.env` file
	load_dotenv()

	# hardware
	device = 'cuda' if torch.cuda.is_available() else 'cpu'

	# load network
	net = torch.load(os.path.join('model_files/', config.DETECTION_MODEL_NAME))

	# data and loader
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

	# mean average precision helper object
	stater = Stater(
		device,
		net,
		config.NUM_CLASSES,
		dataset,
		config.STATS_NUM_ITEMS,
		config.STATS_NUM_REPEATS,
		config.STATS_IOU_THRESHOLDS,
		os.path.join(os.getenv('OUT_DIR'), 'voc_od_stats.png'),
		os.path.join(os.getenv('OUT_DIR'), 'voc_od_map.png'),
		'absolute',
	)
	stater.calculate()
