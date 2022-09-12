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
NUM_CELLS = 8
BOXES_PER_CELL = 2

# model names
BASE_MODEL_NAME = 'voc_od_base.tch'
DETECTION_MODEL_NAME = 'voc_od_detection.tch'

# detection training
D_LEARNING_RATE = 5e-5
D_BATCH_SIZE = 128
D_EPOCHS = 30
VOC_IMAGE_SET = 'train' # 'train', 'val', 'trainval'

# classification training
C_LEARNING_RATE = 1e-5
C_BATCH_SIZE = 256
C_EPOCHS = 1

# detection loss
LAMBDA_COORD = 5
LAMBDA_OBJ = 2
LAMBDA_NO_OBJ = 2
LAMBDA_PROB = 1

# inference and statistics
TEST_NUM_ITEMS = 16
TEST_CONFIDENCE_THRESHOLD = 0.5
STATS_NUM_ITEMS = 512
STATS_NUM_REPEATS = 2
STATS_IOU_THRESHOLDS = [i / 10 for i in range(3, 10)]
