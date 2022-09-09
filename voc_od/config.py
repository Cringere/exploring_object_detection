# data
INPUT_SIZE = 256
INPUT_CHANNELS = 3
CLASSES = [
	'car',
	'cow',
	'tvmonitor',
	'train',
	'horse',
	'sofa',
	'chair',
	'dog',
	'bottle',
	'person',
	'motorbike',
	'boat',
	'bird',
	'sheep',
	'cat',
	'aeroplane',
	'bus',
	'diningtable',
	'pottedplant',
	'bicycle',
]
CLASS_TO_INDEX = { c: i for (i, c) in enumerate(CLASSES)}
NUM_CLASSES = len(CLASSES)
NUM_CLASSIFICATION_CLASSES = 1000

# network
NUM_CELLS = 7
BOXES_PER_CELL = 2

# model names
BASE_MODEL_NAME = 'voc_od_base.tch'
DETECTION_MODEL_NAME = 'voc_od_detection.tch'

# training
LEARNING_RATE = 1e-4
CLASSIFICATION_LEARNING_RATE = 1e-3
BATCH_SIZE = 64
CLASSIFICATION_BATCH_SIZE = 256
EPOCHS = 1
CLASSIFICATION_EPOCHS = 1

# loss
LAMBDA_COORD = 10
LAMBDA_OBJ = 1
LAMBDA_NO_OBJ = 2
LAMBDA_PROB = 2

# inference and statistics
TEST_NUM_ITEMS = 32
TEST_CONFIDENCE_THRESHOLD = 0.7
STATS_NUM_ITEMS = 4096
STATS_IOU_THRESHOLDS = [i / 10 for i in range(3, 10)]
