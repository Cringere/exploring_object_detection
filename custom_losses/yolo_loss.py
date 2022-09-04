from typing import List
import torch
import torch.nn as nn

from utils import iou
from utils.sqrt_scale import SqrtScale

class YoloLoss(nn.Module):
	'''
	A variation of the loss from the YOLO-v1 paper
	'''
	def __init__(self, num_classes):
		super().__init__()
		self.num_classes = num_classes

		self.mse = nn.MSELoss(reduction='sum')
		self.sqrt_scale = SqrtScale()
	
	def forward(self, label, pred):
		'''
		input:
			label: (batch, cells, cells, classes + 5)
			pred:  (batch, cells, cells, classes + 5 * b)
		output: tuple of:
			location_loss
			size_loss
			existence_loss
			existence_loss_no_obj
			probability_loss
		'''

		# extract the dimensions
		(batch, cells_a, cells_b, cp5b) = pred.shape
		b = int((cp5b - self.num_classes) // 5)

		# index at which the labels have their probability score
		probability_index = self.num_classes

		# create two booleans tensors indicating which cells contain objects
		# and which don't
		# (batch, cells, cells)
		i_obj = label[..., probability_index] > 0.5
		i_no_obj = label[..., probability_index] != True

		# split the labels into two groups based on whether they indicate an object
		# (obj_count, classes + 5)
		label_obj = label[i_obj]
		# (no_obj_count, classes + 5)
		label_no_obj = label[i_no_obj]

		# do the same with the predictions
		# (obj_count, classes + 5 * b)
		pred_obj = pred[i_obj]
		# (no_obj_count, classes + 5 * b)
		pred_no_obj = pred[i_no_obj]

		# total number of cells with and without objects (across all batches)
		obj_count = label_obj.shape[0]
		_no_obj_count = label_no_obj.shape[0]

		# extract the boxes fro the `label_obj` tensor
		# (obj_count, 5)
		label_obj_boxes = label_obj[..., self.num_classes:]

		# extract the boxes from the `pred_obj` tensor and place every box in its
		# own dimension
		# (obj_count, 5 * b)
		pred_obj_boxes = pred_obj[..., self.num_classes:]
		# (obj_count, b, 5)
		pred_obj_boxes = pred_obj_boxes.view((obj_count, b, 5))

		# match the dimensions of `pred_obj_boxes` with `label_obj_boxes` by
		# un-squeezing and repeating
		# (obj_count, 1, 5)
		label_obj_boxes_repeated = label_obj_boxes.unsqueeze(1)
		# (obj_count, b, 5)
		label_obj_boxes_repeated = label_obj_boxes_repeated.repeat((1, b, 1))

		# calculate the IOUs
		# (obj_count, b)
		ious = iou(label_obj_boxes_repeated, pred_obj_boxes)

		# find the indices of the boxes with the highest IOUs
		# (obj_count,)
		indices = torch.max(ious, dim=-1)[1]

		# collect all the predicted boxes with the highest IOUs
		# goal: collected[i, k] = pred_obj_boxes[i, indices[i], k]
		# the above can be achieved by using the gather function and slightly
		# reshaping the indices tensor
		# actual: collected[i, 0, k] = pred_obj_boxes[i, indices[i, 0, k], k]
		# mid step: indices[i, 0, k] <- indices[i]
		# (obj_count, 1, 1)
		indices = indices.view((-1, 1, 1))
		# (obj_count, 1, 5)
		indices = indices.repeat((1, 1, 5))
		# (obj_count, 1, 5)
		pred_obj_boxes_max = torch.gather(pred_obj_boxes, 1, indices)
		# (obj_count, 5)
		pred_obj_boxes_max = pred_obj_boxes_max.squeeze(1)

		# location loss - x and y components
		location_loss = self.mse(
			label_obj_boxes[..., 1: 3],
			pred_obj_boxes_max[..., 1: 3]
		)

		# size loss - w and h components scaled by square root
		size_loss = self.mse(
			self.sqrt_scale(label_obj_boxes[..., 3: 5]),
			self.sqrt_scale(pred_obj_boxes_max[..., 3: 5])
		)

		# probability loss of cells with objects
		prob_loss_obj = self.mse(
			label_obj_boxes[..., 0],
			pred_obj_boxes_max[..., 0]
		)

		# probability loss of cells without objects
		# since the number of boxes per cell is very small (probably <5)
		# it is ok to use a for loop here
		prob_loss_no_obj = sum(
			self.mse(
				label_no_obj[..., probability_index],
				pred_no_obj[..., probability_index + 5 * i]
			)
			for i in range(b)
		)
		
		# classes loss for cells with objects
		classes_loss = self.mse(
			label_obj[..., :self.num_classes],
			pred_obj[..., :self.num_classes]
		)

		# scale all losses and return
		m = batch * cells_a * cells_b
		c = self.num_classes
		return (
			(1 / (m + b)) * location_loss,
			(1 / (m + b)) * size_loss,
			(1 / (m + b)) * prob_loss_obj,
			(1 / (m + b)) * prob_loss_no_obj,
			(1 / (m + c)) * classes_loss,
		)
