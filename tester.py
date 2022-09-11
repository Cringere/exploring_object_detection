import torch.nn as nn
from torch.utils.data import DataLoader

from utils.drawer import Drawer
from utils import relative_box_to_absolute

class Tester:
	def __init__(
			self,
			net,
			dataset,
			num_classes,
			cells,
			input_size,
			num_items,
			confidence_threshold,
			out_path,
			unnormalize_fn=lambda x: x,
			device='cpu',
			thickness=2):
		self.net = net
		self.device = device
		self.dataset = dataset
		self.num_classes = num_classes
		self.cells = cells
		self.input_size = input_size
		self.confidence_threshold = confidence_threshold
		self.out_path = out_path
		self.unnormalize_fn = unnormalize_fn
		self.thickness = thickness

		net = net.to(self.device)

		self.dataloader = DataLoader(
			dataset=self.dataset,
			batch_size=num_items,
			shuffle=True,
		)

	def test(self):
		drawers = []
		for imgs, labels in self.dataloader:
			preds = self.net(imgs.to(self.device)).cpu()
			for item in range(imgs.shape[0]):
				img = imgs[item].numpy()
				label = labels[item].numpy()
				pred = preds[item].numpy()

				img = self.unnormalize_fn(img)

				drawer = self.test_item(img, label, pred)
				drawer.border((1, 1, 1))
				drawers.append(drawer)
			break
				
		Drawer.concat_to_grid(drawers, columns=4).save(self.out_path)

	def test_item(self, img, label, pred):
		drawer = Drawer.from_array_chw(img)

		# draw real bounding boxes
		for i in range(label.shape[0]):
			for j in range(label.shape[1]):
				l = label[i, j] # (classes, 5)
				self.draw_bounding_box(i, j, drawer, l, (1, 1, 1))

		# draw predicted bounding boxes
		for i in range(pred.shape[0]):
			for j in range(pred.shape[1]):
				l = pred[i, j].tolist() # (classes + 5 * b)
				classes = l[:self.num_classes]
				boxes = l[self.num_classes:]
				for b in range(len(boxes) // 5):
					box_l = classes + boxes[b * 5: (b + 1) * 5]
					p = max(min(boxes[b * 5], 1.0), 0.0)
					color = (0.2, 1.0, 0.2)
					color = list(map(lambda x: x * p, color))
					self.draw_bounding_box(i, j, drawer, box_l, color)

		return drawer

	def draw_bounding_box(self, i, j, drawer, label, color):
		p = label[self.num_classes]
		if p > self.confidence_threshold:
			box = label[self.num_classes:]
			tl_x, tl_y, br_x, br_y = relative_box_to_absolute(
				i,
				j,
				box[1],
				box[2],
				box[3],
				box[4],
				self.cells,
				self.input_size,
			)

			drawer.bounding_box_from_corners(
				tl_x,
				tl_y,
				br_x,
				br_y,
				color,
				self.thickness
			)
