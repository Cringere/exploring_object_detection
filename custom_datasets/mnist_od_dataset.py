import torch
import torchvision
import torchvision.transforms as T

from torch.utils.data import Dataset, DataLoader

from skimage import transform as ski_transform

import numpy as np

class _RandomResizeRotate:
	def __init__(self, size, degrees):
		self.min_size = size[0]
		self.max_size = size[1]
		self.min_degrees = degrees[0]
		self.max_degrees = degrees[1]
		self.to_tensor = T.ToTensor()
	
	def __call__(self, img):
		# load image (28, 28)
		img = np.array(img)
		
		# resize to (s, s)
		s = np.random.randint(self.min_size, self.max_size)
		img = ski_transform.resize(img, (s, s))
		
		# rotate
		r = np.random.randint(self.min_degrees, self.max_degrees)
		img = ski_transform.rotate(img, r, resize=True)

		# convert to tensor
		# permute to (1, s, s)
		img = self.to_tensor(img)
		return img

class MnistOdDataset(Dataset):
	def __init__(self,
			root,
			train,
			download,
			n_cells,
			single_object_per_cell=True,
			out_size=128,
			dataset_size=4096,
			items_per_image=(2, 5),
			items_rotation=(-30, 30),
			items_size=(25, 35),
			):
		"""
		Generates images with shape (1, out_size, out_size) with scattered mnist
		digits.
		The arrangement of the digits can be controlled with the `items_*`parameters.

		The generated labels follow yolo's style
		"""
		
		self.single_object_per_cell = single_object_per_cell

		self.n_cells = n_cells
		self.out_size = out_size
		self.dataset_size = dataset_size
		self.items_per_image = items_per_image
		self.items_rotation = items_rotation
		self.items_size = items_size

		# create the transform
		transform = T.Compose([
			_RandomResizeRotate(items_size, items_rotation),
		])

		# create the underlying dataset
		self.mnist = torchvision.datasets.MNIST(
			root,
			train,
			download=download,
			transform=transform,
		)

		# create the dataloader
		self.loader = DataLoader(
			self.mnist,
			batch_size=1,
			shuffle=True,
		)

	def __len__(self):
		return self.dataset_size

	def _prepare_img_with_raw_labels(self):
		'''
		Constructs an image out of multiple mnist samples, and returns a list
		with the corresponding bounding boxes.
		returns:
			out_image - tensor of shape (1, self.out_size, self.out_size)
			labels - two dimensional array of the bounding boxes.
				labels[i] - label of bounding box i
				label format (dictionary):
					{
						class: int,
						center: (int, int),
						size: (int, int)
					}
		'''
		
		out_img = torch.zeros(1, self.out_size, self.out_size)
		labels = []
		n = np.random.randint(self.items_per_image[0], self.items_per_image[1] + 1)
		
		if self.single_object_per_cell:
			assert n <= self.n_cells ** 2

		if self.single_object_per_cell:
			available_cells = [
				(i, j)
				for i in range(self.n_cells)
				for j in range(self.n_cells)
			]
			np.random.shuffle(available_cells)

		cell_size = self.out_size / self.n_cells

		for _ in range(n):
			for img, label in self.loader:
				# extract the label
				label_class = label.item()

				# extract the image and the dimensions
				img = img[0] # (1, s, s)
				s = img.shape[1]

				if self.single_object_per_cell:
					# chose a cell
					(i, j) = available_cells.pop()

					# chose a random center inside the cell					
					cr = np.random.uniform(high=cell_size)
					cc = np.random.uniform(high=cell_size)

					# offset to the cell's position
					cr += i * cell_size
					cc += j * cell_size

					# calculate top left corner
					r = int(cr - s // 2)
					c = int(cc - s // 2)
				else:
					# chose a top left corner location
					r = np.random.randint(0, self.out_size - s)
					c = np.random.randint(0, self.out_size - s)

				# if the character is placed out of bounds, bring it back
				r = max(r, 0)
				c = max(c, 0)
				if r + s >= self.out_size:
					r = self.out_size - s
				if c + s >= self.out_size:
					c = self.out_size - s

				# add to the complete image
				out_img[:, r: r + s, c: c + s] += img

				# clamp pixel values
				out_img.clamp(0, 1)

				# construct a label
				labels.append({
					'center': (r + s // 2, c + s // 2),
					'size': (s, s),
					'class': label_class
				})
				break
		
		# pixels' color must be valid
		out_img.clamp_(0.0, 1.0)
	
		return out_img, labels

	def _process_raw_labels(self, labels):
		'''
		inputs:
			labels - labels generated with the function `_prepare_img_with_raw_labels`
		outputs:
			grid_labels - input labels transformed into a grid.
				shape: (num_cells, num_cells, 10 + 1 + 2 + 2).
				cells that don't contain an object's center point, will have a
				vector of zeros (the important part is that p will be 0)
				each cell has the following format
				[c1 .. c10, p, cx, cy, w, h]
					where:
						c1 ... c10 - one hot corresponding to the class
						p - the probability of the class
							- 1 if it contains an object's center and 0 otherwise
						cx, cy - center x and y (between 0 and 1)
						w, h - width and height (relative to the size of the cell)
		'''

		# calculate the size of a cell
		cell_width = self.out_size / self.n_cells
		cell_height = self.out_size / self.n_cells

		# iterate over the labels, adding them to the grid, and transforming
		# their coordinates to be relative to a target cell
		grid_labels = torch.zeros((self.n_cells, self.n_cells, 10 + 1 + 2 + 2)).float()
		for label in labels:
			# extract the cell's center
			center_row, center_col = label['center']

			# calculate the corresponding cell
			cell_row = int(center_row // cell_height)
			cell_col = int(center_col // cell_width)

			# extract the label class
			label_class = int(label['class'])

			# normalize the object's center
			y = (center_row - (cell_row * cell_height)) / cell_height
			x = (center_col - (cell_col * cell_width)) / cell_width

			# normalize the object's size
			w, h = label['size']
			w /= cell_width
			h /= cell_height
			
			# construct a grid label
			grid_labels[cell_row][cell_col][label_class] = 1.0
			grid_labels[cell_row][cell_col][10] = 1.0
			grid_labels[cell_row][cell_col][11:] = torch.Tensor([
				x, y, w, h
			])

		return grid_labels

	def __getitem__(self, _):
		img, raw_labels = self._prepare_img_with_raw_labels()
		labels = self._process_raw_labels(raw_labels)
		return img, labels

	def relative_box_to_absolute(self, cell_row, cell_col, cx, cy, w, h):
		"""
		input:
			cell_row, cell_col - the cell that the bounding box belongs to
			cx, cy - bounding box's center relative to the the cell
			w, h - bounding box's size relative to the cell
		output:
			r, c - row and column (in pixels) of the top left corner of the box
			w, h - size (in pixels) of the the box
		"""
		# cell's size
		cell_width = self.out_size / self.n_cells
		cell_height = self.out_size / self.n_cells

		# un-normalize size
		w = int(w * cell_width)
		h = int(h * cell_height)

		# un-normalize the offset
		ox = cx * cell_width
		oy = cy * cell_height

		# get the cell's position
		cell_x = cell_width * cell_col
		cell_y = cell_height * cell_row

		# box's position
		cx = int(cell_x + ox)
		cy = int(cell_y + oy)

		return (
			cx - w // 2,
			cy - h // 2,
			cx + w // 2,
			cy + h // 2,
		)

if __name__ == '__main__':
	import os
	from dotenv import load_dotenv
	load_dotenv()

	dataset = MnistOdDataset(
		root=os.getenv('MNIST_ROOT'),
		train=True,
		download=True,
		n_cells=5,
	)
	loader = DataLoader(dataset, batch_size=1, shuffle=True)

	for img, labels in loader:
		img = img[0].numpy()
		labels = labels[0].numpy()

		img = img.transpose((1, 2, 0)) # permute to (s, s, 1)

		from skimage.draw import polygon_perimeter

		# draw bounding boxes
		for i in range(labels.shape[0]):
			for j in range(labels.shape[1]):
				label = labels[i, j]
				classes = label[: 10]
				p = label[10]
				center = label[11: 11 + 2]
				size = label[13: 13 + 2]
				if p > 0.8:
					tl_x, tl_y, br_x, br_y = dataset.relative_box_to_absolute(
						i,
						j,
						center[0],
						center[1],
						size[0],
						size[1]
					)
					
					r = [tl_y, br_y, br_y, tl_y]
					c = [tl_x, tl_x, br_x, br_x]
					rr, cc = polygon_perimeter(r, c, img.shape)
					img[rr, cc] = 1
		
		path = os.path.join(os.getenv('OUT_SAMPLES_DIR'), 'mnist_od_data_sample.png')
		
		from skimage.io import imsave
		imsave(path, img)

		break
