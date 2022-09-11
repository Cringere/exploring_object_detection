import random
from custom_losses.yolo_loss import YoloLoss

from utils import iou

import torch

from math import pow, sqrt

def manual_iou(ax, ay, aw, ah, bx, by, bw, bh):
	delta_x = max(
		min(ax + aw / 2, bx + bw / 2) -
		max(ax - aw / 2, bx - bw / 2),
		0
	)
	
	delta_y = max(
		min(ay + ah / 2, by + bh / 2) -
		max(ay - ah / 2, by - bh / 2),
		0
	)

	intersection = delta_x * delta_y
	union = aw * ah + bw * bh - intersection

	return intersection / (union + 1e-6)

if __name__ == '__main__':
	# global properties
	img_size = (128, 128)
	classes = 3
	cells = 2
	s = 64

	# helpers
	def one_hot(index):
		arr = [0] * classes
		arr[index] = 1
		return arr
	
	def empty_cell_label():
		return [0] * (classes + 5)

	def random_cell_prediction():
		return [random.random() for _ in range(classes + 5 * 2)]

	# first target
	classes_1 = one_hot(0)
	p1 = (28 / s, 28 / s)
	s1 = (32 / s, 32 / s)

	# first target predicted classes
	c1 = [0.8, 0.1, 0.2]

	# first bounding box of first target
	b_prob_11 = 0.5
	b_p11 = (42 / s, 42 / s)
	b_s11 = (30 / s, 30 / s)

	# second bounding box of first target
	b_prob_12 = 1.0
	b_p12 = (34 / s, 5 / s)
	b_s12 = (20 / s, 25 / s)

	# second target
	classes_2 = one_hot(1)
	p2 = ((110 - s) / s, (75 - s) / s)
	s2 = (20 / s, 40 / s)

	# second target predicted classes
	c2 = [0.8, 0.1, 0.2]

	# first bounding box of second target
	b_prob_21 = 0.1
	b_p21 = ((80 - s) / s, (66 - s) / s)
	b_s21 = (20 / s, 20 / s)

	# second bounding box of second target
	b_prob_22 = 0.2
	b_p22 = ((110 - s) / s, (100 - s) / s)
	b_s22 = (40 / s, 30 / s)

	label_11 = classes_1 + [1] + [p1[0], p1[1], s1[0], s1[1]]
	label_22 = classes_2 + [1] + [p2[0], p2[1], s2[0], s2[1]]

	label = [
		[ # batch
			[ # row
				label_11,
				# column
				empty_cell_label(),
			],
			[ # row
				# column
				empty_cell_label(),
				# column
				label_22,
			],
		]
	]

	pred_11 = (
		# classes
		c1 +
		# first box
		[b_prob_11] +
		[b_p11[0], b_p11[1], b_s11[0], b_s11[1]] +
		# second box
		[b_prob_12] +
		[b_p12[0], b_p12[1], b_s12[0], b_s12[1]])
	
	pred_22 = (
		# classes
		c2 +
		# first box
		[b_prob_21] +
		[b_p21[0], b_p21[1], b_s21[0], b_s21[1]] +
		# second box
		[b_prob_22] +
		[b_p22[0], b_p22[1], b_s22[0], b_s22[1]])

	pred = [
		[ # batch
			[ # row
				pred_11,
				random_cell_prediction(),
			],
			[ # row
				random_cell_prediction(),
				pred_22
			],
		]
	]

	#  0  1  2  3  4  5  6  7  8  9  10  11  12
	#  c1 c2 c3 p  x  y  w  h  p  x  y   w   h
	ious = [
		iou(torch.Tensor(label_11[3: 8]), torch.Tensor(pred_11[3: 8])),  # 0.176
		iou(torch.Tensor(label_11[3: 8]), torch.Tensor(pred_11[8: 13])), # 0.077
		iou(torch.Tensor(label_22[3: 8]), torch.Tensor(pred_22[3: 8])),  # 0.0
		iou(torch.Tensor(label_22[3: 8]), torch.Tensor(pred_22[8: 13])), # 0.111
	]

	# print(ious)
	
	# print(iou(torch.Tensor(label_22[3: 8]), torch.Tensor(pred_22[8: 13])))
	# print(label_22[3: 8])
	# print(pred_22[8: 13])
	# print(manual_iou(
	# 	0.7, 0.1, 0.3, 0.6,
	# 	0.7, 0.5, 0.3, 0.1,
	# ))

	label = torch.Tensor(label)
	pred = torch.Tensor(pred)
	# print(label.shape)
	# print(pred.shape)

	loss_fn = YoloLoss(3, 'cpu')
	print(loss_fn(label, pred))

	print(
		(1 / 4) *
		(
			pow(p1[0] - b_p11[0], 2) + 
			pow(p1[1] - b_p11[1], 2) + 
			pow(p2[0] - b_p22[0], 2) + 
			pow(p2[1] - b_p22[1], 2)
		)
	)

	print(
		(1 / 4) *
		(
			pow(sqrt(s1[0]) - sqrt(b_s11[0]), 2) + 
			pow(sqrt(s1[1]) - sqrt(b_s11[1]), 2) + 
			pow(sqrt(s2[0]) - sqrt(b_s22[0]), 2) + 
			pow(sqrt(s2[1]) - sqrt(b_s22[1]), 2)
		)
	)

	print(
		(1 / 6.0) *
		(
			pow(pred_11[0] - label_11[0], 2) +
			pow(pred_11[1] - label_11[1], 2) +
			pow(pred_11[2] - label_11[2], 2) +

			pow(pred_22[0] - label_22[0], 2) +
			pow(pred_22[1] - label_22[1], 2) +
			pow(pred_22[2] - label_22[2], 2)
		)
	)

	print('todo')

	print(
		(1 / 6.0) *
		(
			pow(b_prob_11 * pred_11[0] - 1 * label_11[0], 2) +
			pow(b_prob_11 * pred_11[1] - 1 * label_11[1], 2) +
			pow(b_prob_11 * pred_11[2] - 1 * label_11[2], 2) +

			pow(b_prob_22 * pred_22[0] - 1 * label_22[0], 2) +
			pow(b_prob_22 * pred_22[1] - 1 * label_22[1], 2) +
			pow(b_prob_22 * pred_22[2] - 1 * label_22[2], 2)
		)
	)
