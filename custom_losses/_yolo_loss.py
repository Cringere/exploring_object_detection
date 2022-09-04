import torch
import torch.nn as nn

# import utils.iou as iou
from utils.iou import iou

class YoloLoss(nn.Module):
	'''
	My initial implementation of a variation of the YOLO loss.
	I tried improving it with the `yolo_loss.py` file.
	'''
	
	def __init__(self, num_classes, boxes_per_cell):
		super().__init__()
		self.num_classes = num_classes
		self.boxes_per_cell = boxes_per_cell

		self.mse = nn.MSELoss(reduction='sum')
	
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
		(batch, cells_a, cells_b) = label.shape[0: 3]

		# extract the predicted classes
		pred_classes = pred[:, :, :, :self.num_classes]

		# extract the boxes
		# (batch, cells, cells, 5)
		label_boxes = label[:, :, :, self.num_classes:]
		# (batch, cells, cells, 5 * b)
		pred_boxes = pred[:, :, :, self.num_classes:]

		# extend the label boxes to match the shape of the predicted boxes
		# (batch, cells, cells, 5 * b)
		label_boxes = label_boxes.repeat((1, 1, 1, self.boxes_per_cell))

		# pull the box index to a new dimension
		# (batch, cells, cells, b, 5)
		(batch, cells_a, cells_b, five_times_b) = label_boxes.shape
		target_shape = (batch, cells_a, cells_b, five_times_b // 5, 5)
		label_boxes = label_boxes.view(target_shape)
		pred_boxes = pred_boxes.view(target_shape)

		# perform IOU
		# (batch, cells, cells, b)
		boxes_iou = iou(label_boxes, pred_boxes)

		# find the bounding with the highest iou
		# (batch, cells, cells)
		(_, indices) = torch.max(boxes_iou, dim=-1)

		# put the index at each own dimension
		# (batch, cells, cells, 1)
		indices = indices.unsqueeze(-1)

		# index the predicted boxes with the indices and select targe box
		# (batch, cells, cells, b, 5) -> (batch, cells, cells, 5)
		# the goal is to perform:
		# out[b, c1, c2, i] = boxes[b, c1, c2, index[b, c1, c2], i]
		# in order to do that, the indices need to be reshaped to
		# (batch, cells, cells, b, 5)
		# first, add another dimension: (batch, cells, cells, 1, 1)
		indices = indices.unsqueeze(-1)
		# then, repeat the boxes dimension: (batch, cells, cells, 1, 5)
		indices = indices.repeat(1, 1, 1, 1, 5)
		# perform the indexing:
		# target_boxes[batch, c1, c2, i] =
		# 	pred_boxes[batch, c2, c2, index[b, c1, c2, i], i]
		# out shape: (batch, cells, cells, 1, 5)
		target_boxes = torch.gather(pred_boxes, 3, indices)
		# remove the extra dimension (batch, cells, cells, 5)
		target_boxes = target_boxes.squeeze(3)


		# object identity - True if a cell has an object's center, and
		# False otherwise
		# (batch, cells, cells)
		i_obj = label[..., self.num_classes] > 0.5
		i_no_obj = i_obj != True
		
		# extract the cells that contain objects
		# we don't know what x is because it depends on the specific
		# data item
		# (x, classes + 5)
		filtered_label = label[i_obj]
		# (x, 5)
		filtered_label_boxes = filtered_label[..., self.num_classes:]
		# (x, classes)
		filtered_label_classes = filtered_label[..., :self.num_classes]
		# (x, 5)
		filtered_target_boxes = target_boxes[i_obj]
		# (x, classes)
		filtered_pred_classes = pred_classes[i_obj]

		# extract the classes for the cells that don't have objects
		# (y, classes)
		filtered_label_boxes_no_obj = label[i_no_obj][..., self.num_classes:]
		filtered_target_boxes_no_obj = target_boxes[i_no_obj]

		# location loss - comparing coordinates
		location_loss = self.mse(
			filtered_label_boxes[:, 1: 3], filtered_target_boxes[:, 1: 3]
		)

		# size loss - comparing with and height
		size_loss = self.mse(
			torch.sign(filtered_label_boxes[:, 3: 5]) * torch.sqrt(torch.abs(filtered_label_boxes[:, 3: 5]) + 1e-4),
			torch.sign(filtered_target_boxes[:, 3: 5]) * torch.sqrt(torch.abs(filtered_target_boxes[:, 3: 5]) + 1e-4)
		)

		# existence loss - comparing the probability of a cell having an object
		existence_loss = self.mse(filtered_target_boxes[:, 0: 1], filtered_label_boxes[:, 0: 1])

		# existence loss - comparing the probability of a cell having an object
		# for cells that don't have objects. This loss should be trained on all
		# the boxes, not just the ones with the highest iou - since if there is
		# no object, the iou of all th boxes is 0
		existence_loss_no_obj = []
		for b in range(self.boxes_per_cell):
			existence_loss_no_obj.append(self.mse(
				label[..., 0], # (batch, cells, cells) <- tensor of 0s
				pred_boxes[..., b, 0] # (batch, cells, cells) <- should be 0s
			))
		existence_loss_no_obj = sum(existence_loss_no_obj)

		# classes probabilities loss - scalar
		probability_loss = self.mse(filtered_label_classes, filtered_pred_classes)

		m = 1.0 / (batch * cells_a * cells_b)

		return (
			m * location_loss,
			m * size_loss,
			m * existence_loss,
			m * existence_loss_no_obj,
			m * probability_loss,
		)
