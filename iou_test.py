import torch

from utils import iou, relative_boxes_to_absolute_tensors

if __name__ == '__main__':
	a = torch.Tensor([
		# p, x, y, w, h
		[1, 1, 1, 2, 2]
	]).float()
	b = torch.Tensor([
		# p, x, y, w, h
		[1, 0, 0, 1, 1]
	]).float()
	print(iou(a, b)) # expected 0.25 / (0.75 + 4) = 0.052

	# boxes: (batch, row cells, colum cells, 5)
	relative_boxes = torch.Tensor([
		[
			[
				[1.0, 0.5, 0.5, 1.0, 1.0],
				[1.0, 0.5, 0.5, 1.0, 1.0],
			],
			[
				[1.0, 0.5, 0.5, 1.0, 1.0],
				[1.0, 0.5, 0.5, 1.0, 1.0],
			],
		]
	]).float()
	print(relative_boxes.shape)
	print(relative_boxes_to_absolute_tensors(relative_boxes))
