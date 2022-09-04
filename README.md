## Exploring Object detection methods
This is a collection of test relating to object detection.

This repository accommodates this [article](https://royvaron.com) which contains
explanations about the algorithms and networks used.

## .env
Before running any of the main files, a `.env.` needs to be present. Here is its
specifications:
```
# data
MNIST_ROOT=/path/to/mnist/
MNIST_DOWNLOAD=True # True / False

# output
OUT_DIR=./out

# model
LOAD_MODEL=False # True / False
SAVE_MODEL=True  # True / False
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
