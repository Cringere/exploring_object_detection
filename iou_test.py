import torch

from utils import iou

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
