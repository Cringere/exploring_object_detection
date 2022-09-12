import torch
import torch.nn as nn
from typing import List

def iou(boxesA, boxesB):
	'''
	input: two tensors of shape (*, 5).
		the last 5 elements correspond to (p, x, y, w, h)
	output: tensor of shape (*) with the iou of all the paired boxes
	'''
	# extract a's coordinates
	ax = boxesA[..., 1]
	ay = boxesA[..., 2]
	aw = boxesA[..., 3]
	ah = boxesA[..., 4]

	# extract b's coordinates
	bx = boxesB[..., 1]
	by = boxesB[..., 2]
	bw = boxesB[..., 3]
	bh = boxesB[..., 4]

	# x segment shared between the boxes
	delta_x = (
		torch.minimum(ax + aw / 2, bx + bw / 2) -
		torch.maximum(ax - aw / 2, bx - bw / 2)
	).clamp(0.0, None)

	# y segment shared between the boxes
	delta_y = (
			torch.minimum(ay + ah / 2, by + bh / 2) -
			torch.maximum(ay - ah / 2, by - bh / 2)
	).clamp(0.0, None)

	# intersection and union
	intersection = delta_x * delta_y
	union = aw * ah + bw * bh - intersection

	# iou with numerical stability
	return intersection / (union + 1e-5)

def iou_c_stats(
		num_classes,
		label,
		pred,
		iou_thresholds: List[float]):
	'''
	Calculates precision and recall for all available confidence thresholds for
	the given IOUs. Assumes true positive predictions share the same cell as their
	target labels.

	input: Tensors
		label: (batch, cells, cells, classes + 5)
		pred:  (batch, cells, cells, classes + 5 * b)
	output:
		iou_c_stats - iou_c_stats[iou_threshold][class] is a map of
			{
				'p': [...], # precision
				'r': [...], # recall
				't': [...], # confidence threshold
			}
	'''
	with torch.no_grad():
		# extract shape
		(batch, cells_a, cells_b, cp5b) = pred.shape
		b = (cp5b - num_classes) // 5

		# flatten
		# (batch * cells * cells, classes + 5) = (x, classes + 5)
		label = label.view((batch * cells_a * cells_b, -1))
		# (batch * cells * cells, classes + 5 * b) = (x, classes + 5 * b)
		pred = pred.reshape((batch * cells_a * cells_b, -1))

		# extract the classes
		# (x, classes)
		label_classes = label[..., :num_classes]
		pred_classes = pred[..., :num_classes]

		# extract the label's boxes
		# (x, 5)
		label_boxes = label[..., num_classes:]

		# extract the prediction's boxes and put them in their own dimension
		# (x, b, 5)
		pred_boxes = pred[..., num_classes:].reshape((-1, b, 5))

		# extract the probabilities
		# (x, b)
		pred_boxes_probs = pred_boxes[..., 0]

		# chose the box with the maximum probability
		# (x, 1)
		(_, indices) = torch.max(pred_boxes_probs, dim=1, keepdim=True)
		# reshape to match `pred_boxes`
		indices = indices.unsqueeze(1).repeat(1, 1, 5)
		# gather the boxes
		# pred_boxes[i, 0, k] <- pred_boxes[i, indices[i, 0, k], k]
		# (x, 1, 5)
		pred_boxes = torch.gather(pred_boxes, 1, indices)
		# remove the extra dimension
		# (x, 5)
		pred_boxes = pred_boxes.squeeze(1)

		# calculate the IOUs for all the boxes with their targets
		# (x,)
		ious = iou(
			label_boxes,
			pred_boxes
		)

		# helper class for storing groups of tensors
		class DataTuple:
			def __init__(
					self,
					label_class,
					pred_class,
					label_box,
					pred_box,
					label_pred_iou):
				self.label_class = label_class
				self.pred_class = pred_class
				self.label_box = label_box
				self.pred_box = pred_box
				self.iou = label_pred_iou
				self.predicted_class = torch.argmax(pred_class).item()
			
			def is_of_class(self, c):
				'''
				return true if the prediction predicted class `c`
				'''
				return self.predicted_class == c
			
			def is_true_positive(self, iou_threshold):
				'''
				label and predicted indicate an existence of an object
				label and predicted iou is high enough.
				assumes predicted probability is sufficient.
				'''
				return self.is_label_positive() and self.iou > iou_threshold
			
			def is_label_positive(self):
				return self.label_box[0] > 0.5

		# create testing tuples
		data = []
		for i in range(ious.shape[0]):
			data.append(DataTuple(
				label_classes[i],
				pred_classes[i],
				label_boxes[i],
				pred_boxes[i],
				ious[i],
			))
		
		# sort by the probability of the predicted box - highest to lowest
		data.sort(key=lambda dt: dt.pred_box[0], reverse=True)

		# map of precision and recall for each class for each iou threshold
		iou_c_stats = {}
		for iou_thresh in iou_thresholds:
			iou_c_stats[iou_thresh] = {}
			for c in range(num_classes):
				iou_c_stats[iou_thresh][c] = {}

		# create the map
		for iou_thresh in iou_thresholds:
			for c in range(num_classes):
				# filter the data with the target class
				data_c = list(filter(lambda dt: dt.is_of_class(c), data))

				total_positive_labels = sum(
					(1 if dt.is_label_positive() else 0)
					for dt in data_c
				)

				precisions = [] # TP / (TP + FP) = TP / (all positive predictions)
				recalls = []    # TP / (TP + FN) = TP / (all positive labels)

				# confidence threshold for which the above stats
				# are calculated
				thresholds = []

				# iterate over all the data, populating the precision and recall
				# arrays
				total_positive_guesses = 0
				tps = 0 # true positives
				for dt in data_c:
					if dt.is_true_positive(iou_thresh):
						tps += 1
					
					total_positive_guesses += 1

					precisions.append(tps / total_positive_guesses)
					recalls.append(tps / total_positive_labels)
					thresholds.append(dt.pred_box[0])

				iou_c_stats[iou_thresh][c] = {
					'p': precisions,
					'r': recalls,
					't': thresholds,
				}
	
	return iou_c_stats

def mean_average_precision(stats):
	'''
	input:
		stats: output of `iou_c_stats`
	output:
		ious: list of ious
		m_ap: list of mean average precisions corresponding to
			the ious
	'''

	def area(xs, ys):
		assert len(xs) == len(ys)
		a = 0
		for i in range(len(xs) - 1):
			delta_x = xs[i + 1] - xs[i]
			y2 = ys[i + 1]
			y1 = ys[i]
			a += delta_x / 2 * (3 * y2 - y1)
		return a
	
	ious = list(stats.keys())
	classes = list(stats[ious[0]].keys())

	m_ap = []
	for iou in ious:
		ap = 0
		for c in classes:
			m = stats[iou][c]
			p = m['p']
			r = m['r']
			a = area(r, p)
			ap += a
		m_ap.append(ap / len(classes))
	
	return ious, m_ap

def count_parameters(module: nn.Module):
	'''
	counts and prints how many parameters a module has
	'''

	n_params = sum(p.numel() for p in module.parameters() if p.requires_grad)
	n_params = "{:,}".format(n_params)
	print(f'model has {n_params} learnable parameters')

def relative_box_to_absolute(cell_row, cell_col, cx, cy, w, h, cells, size):
		'''
		input:
			cell_row, cell_col - the cell that the bounding box belongs to
			cx, cy - bounding box's center relative to the the cell
			w, h - bounding box's size relative to the cell
			cells - the number of cells the width and height of the original image
			        was split to
			size - number of rows and columns in pixels
		output:
			r, c - row and column (in pixels) of the top left corner of the box
			w, h - size (in pixels) of the the box
		'''

		if isinstance(cells, int):
			cells = (cells, cells)

		if isinstance(size, int):
			size = (size, size)

		# cell size
		cell_width = size[1] / cells[1]
		cell_height = size[0] / cells[0]

		# un-normalize size
		w = w * cell_width
		h = h * cell_height

		# get the cell's position
		cx = cell_col * cell_width + cx * cell_width
		cy = cell_row * cell_height + cy * cell_height

		return (
			int(cx - w / 2),
			int(cy - h / 2),
			int(cx + w / 2),
			int(cy + h / 2),
		)
