import torch

from typing import List, Dict

class LabelObject:
	def __init__(
			self,
			name: str,
			xmin: int,
			xmax: int,
			ymin: int,
			ymax: int,
			difficult: bool,
			occluded: bool,
			pose: str,
			truncated: bool,
		):
		self.name = name
		self.xmin = xmin
		self.xmax = xmax
		self.ymin = ymin
		self.ymax = ymax
		self.difficult = difficult
		self.occluded = occluded
		self.pose = pose
		self.truncated = truncated
	
	def normalized(self, w: float, h: float) -> List[float]:
		'''
		returns (xmin, xmax, ymin ymax) normalized to be be between 0 and 1
		'''
		return [
			self.xmin / w,
			self.xmax / w,
			self.ymin / h,
			self.ymax / h,
		]

class Source:
	def __init__(
			self,
			annotation: str,
			database: str,
			image: str,
			):
		self.annotation = annotation
		self.database = database
		self.image = image

class VocLabel:
	def __init__(
			self,
			folder: str,
			filename: str,
			source: Source,
			size: List[int],
			segmented: bool,
			label_objects: List[LabelObject]
			):
		'''
		folder: the folder of the image
		filename: the name of the image
		source: where the image is taken from
		size: list: (channels, height, width)
		segmented: TODO
		label_objects: a list of objects present int the image
		'''

		self.folder = folder
		self.filename = filename
		self.source = source
		self.size = size
		self.segmented = segmented
		self.label_objects = label_objects
	
	def to_tensor(self, cells: int, classes: Dict[str, int]) -> torch.Tensor:
		'''
		creates a training label from the current label.
		The output is a tensor shaped (cells, cells, n_classes + 5)
		'''
		n_classes = len(classes)
		cell_size = 1.0 / cells
		l = torch.zeros((cells, cells, n_classes + 5))
		for obj in self.label_objects:
			# normalize coordinates
			xmin, xmax, ymin, ymax = obj.normalized(self.size[2], self.size[1])
			center_x = (xmin + xmax) / 2
			center_y = (ymin + ymax) / 2

			# find the cell
			cell_row = int(center_y / cell_size)
			cell_col = int(center_x / cell_size)

			# encode the class
			l[cell_row, cell_col, classes[obj.name]] = 1

			# encode the probability and coordinates
			l[cell_row, cell_col, n_classes] = torch.Tensor([
				# probability
				1.0,
				# coordinates
				(center_x - cell_col * cell_size) / cell_size,
				(center_y - cell_row * cell_size) / cell_size,
				# size
				(xmax - xmin) / cell_size,
				(ymax - ymin) / cell_size,
			])
		
		return l

	@classmethod
	def from_annotation(cls, annotation):
		# get the sub objects from the annotation
		folder = annotation['folder']
		filename = annotation['filename']
		source = annotation['source']
		size = annotation['size']
		segmented = annotation['segmented']
		object = annotation['object']

		# parse the sub objects into a label objects
		return VocLabel(
			folder,
			filename,
			Source(
				source['annotation'],
				source['database'],
				source['image'],
			),
			[
				int(size['depth']),
				int(size['height']),
				int(size['width'])
			],
			segmented == '1',
			[LabelObject(
				obj['name'],
				int(obj['bndbox']['xmin']),
				int(obj['bndbox']['xmax']),
				int(obj['bndbox']['ymin']),
				int(obj['bndbox']['ymax']),
				obj['difficult'] == '1',
				obj['occluded'] == '1',
				obj['pose'],
				obj['truncated'] == '1',
			) for obj in object]
		)
	
	@classmethod
	def from_label(cls, label):
		'''
		label: the target of an item from torchvision.datasets.VOCDetection
		'''
		return cls.from_annotation(label['annotation'])
