## Exploring Object detection methods
This is a collection of test relating to object detection.

This repository accommodates this [article](https://royvaron.com) which contains
explanations about the algorithms and networks used.

## .env
Before running any of the main files, a `.env.` needs to be present. Here is an example:
```
# mnist
MNIST_ROOT=/path/to/mnist/
MNIST_DOWNLOAD=False

# voc
VOC_ROOT=/home/cloud/datasets
VOC_DOWNLOAD=False

# imagenet
IMAGE_NET_ROOT=/path/to/image-net/

# data loading
NUM_WORKERS=50

# output
OUT_DIR=./out

# mnist model
MNIST_LOAD_MODEL=False
MNIST_SAVE_MODEL=True

# voc model
# Stages:
#     None - don't save or load.
#     Base - base model.
#     Full - Both the base and the head
VOC_LOAD_STAGE=Full
# comma separated list of the models to save.
# parameters: Base, Full
VOC_SAVE_MODELS=Base, Full

```

## Usage - Mnist
1. Train the model - `python mnist_od_train.py`
	* set `SAVE_MODEL=True` in `.env`.
	* Generated files:
		* `OUT_DIR/mnist_od_losses.png`
		* `OUT_DIR/mnist_od_losses_offset_60.png`
2. Test the model - `python mnist_od_test.py`
	* Generated files"
		* `OUT_DIR/mnist_od_test_sample.png`
3. Extract statistics from the model - `python mnist_od_stats.py`
	* Generated files:
		* `OUT_DIR/mnist_od_stats.png`
		* `OUT_DIR/mnist_od_mean_average_precision.png`

## Usage - VOC
1. Train the model - `python voc_od_train.py`
	* set `SAVE_MODEL=True` in `.env`.
	* Generated files:
		* `OUT_DIR/voc_od_losses.png`
		* `OUT_DIR/voc_od_losses_offset_60.png`
2. Test the model - `python voc_od_test.py`
	* Generated files"
		* `OUT_DIR/voc_od_test_sample.png`
3. Extract statistics from the model - `python voc_od_stats.py`
	* Generated files:
		* `OUT_DIR/voc_od_stats.png`
		* `OUT_DIR/voc_od_mean_average_precision.png`
