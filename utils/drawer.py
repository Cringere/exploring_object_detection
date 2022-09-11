import numpy as np

from skimage.draw import polygon_perimeter

def _convert_color(color):
	assert isinstance(color, (list, tuple, int, float))

	if not isinstance(color, (list, tuple)):
		color = [color, color, color]

	return color

class Drawer:
	'''
	Loading images, saving images, drawing bounding boxes
	'''
	def __init__(self, img):
		'''
		expects a numpy array as an image.
		img: (height (rows), width (columns), channels)
		channels can be either 1 or 3
		'''
		assert img.shape[2] == 1 or img.shape[2] == 3

		if img.shape[2] == 1:
			img = np.repeat(img, 3, axis=2) # repeat to (r, c, 3)
		self.img = img
	
	def bounding_box_from_corners(self, tl_x: int, tl_y: int, br_x: int, br_y: int, color, thickness=1):
		for i in range(thickness):
			color = _convert_color(color)
			
			r = [tl_y - i, br_y + i, br_y + i, tl_y - i]
			c = [tl_x - i, tl_x - i, br_x + i, br_x + i]
			rr, cc = polygon_perimeter(r, c, self.img.shape)
			self.img[rr, cc] = np.maximum(color, self.img[rr, cc])
	
	def border(self, color):
		self.bounding_box_from_corners(0, 0, self.img.shape[1], self.img.shape[0], color)

	def save(self, path):
		from skimage.io import imsave
		imsave(path, self.img)

	@classmethod
	def from_array_chw(cls, img):
		'''
		img: (channels, height (rows), width (columns))
		'''
		return Drawer(img.transpose((1, 2, 0)))

	@classmethod
	def from_array_hwc(cls, img):
		'''
		img: (height (rows), width (columns), channels)
		'''
		return Drawer(img)
	
	@classmethod
	def from_uniform_color(cls, size, color):
		assert isinstance(size, (tuple, list))
		color = _convert_color(color)
		return Drawer(np.full(size, np.array(color)))

	@classmethod
	def concat_to_grid(cls, imgs, columns=1):
		assert len(imgs) > 0
		
		# assert that all images are either numpy arrays or drawers
		# convert all images to numpy arrays
		# find the max size of all the images
		max_size = (0, 0)
		np_imgs = []
		for img in imgs:
			assert isinstance(img, (Drawer, np.ndarray))

			if isinstance(img, Drawer):
				img = img.img

			max_size = (
				max(max_size[0], img.shape[0]),
				max(max_size[1], img.shape[1]),
			)

			np_imgs.append(img)
		
		# add images until the length matches the number of columns
		while len(np_imgs) % columns != 0:
			np_imgs.append(cls.from_uniform_color(max_size, (0, 0, 0)))
		
		# create the rows - concatenate over the first axis
		row_len = len(np_imgs) // columns
		rows = [
			np.concatenate(np_imgs[i * row_len: (i + 1) * row_len], axis=0)
			for i in range(4)
		]

		# concatenate the rows to create the full image
		return Drawer(np.concatenate(rows, axis=1))
