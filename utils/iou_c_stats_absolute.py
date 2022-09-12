import torch
import torch.nn as nn

class Stats:
	def __init__(self, pred_shape, label_shape, iou_thresholds):
		'''
		input:
			pred: (batches, cells, cells, classes + 5)
			label: (batches, cells, cells, classes + 5 * boxes)
			iou_thresholds
		'''
		self.batches, cells_a, cells_b, cp5  = label_shape
		assert cells_a == cells_b
		self.cells = cells_a
		self.classes = cp5 - 5
		self.num_boxes = (pred_shape[-1] - self.classes) // 5
		self.iou_thresholds = iou_thresholds

	def iou_c_stats(self, pred, label):
		# extract the predicted classes
		# (batch, cells, cells, classes)
		pred_classes = pred[..., :self.classes]

		# extract the predicted bounding boxes
		# (batch, cells, cells, 5 * b)
		pred_boxes = pred[..., self.classes:]

		# place the boxes in their own dimensions
		# (batch, cells, cells, b, 5)
		pred_boxes = pred_boxes.view((self.batches, self.cells, self.cells, self.num_boxes, 5))

		# extract the probabilities of the boxes
		# (batch, cells, cells, b)
		pred_probs = pred_boxes[..., 0]

		# find the indices of the maximum probabilities
		# (batch, cells, cells, 1)
		_, indices = torch.max(pred_probs, dim=-1, keepdim=True)

		# reshape the indices to match the boxes
		# (batch, cells, cells, 1, 5)
		indices = indices.unsqueeze(-1).repeat((1, 1, 1, 1, 5))

		# gather the boxes with the maximum indices
		# pred_boxes[i, j, k, 0, l] <- pred_boxes[i, j, k, indices[i, j, k, l], l]
		# pred_boxes[batch, cells, cells, 1, 5]
		pred_boxes = torch.gather(pred_boxes, dim=3, index=indices)

		# remove the boxes dimension
		# pred_boxes[batch, cells, cells, 5]
		pred_boxes = pred_boxes.squeeze(3)
	
		# concat the boxes with the classes
		# pred_boxes[batch, cells, cells, classes + 5]
		pred = torch.concat((pred_classes, pred_boxes), dim=-1)

		# (batch, cells^2, classes + 5)
		pred = Stats.to_absolute(pred)
		label = Stats.to_absolute(label)

		# [(cells^2, classes + 5,)], len = batches
		label = [label[i] for i in range(self.batches)]

		# [(x, classes + 5,)], len = batches
		label = [l[l[:, self.classes] > 0.5] for l in label]

		# [(cells^2, classes + 5,)], len = batches
		pred = [pred[i] for i in range(self.batches)]

		for i in range(self.batches):
			# (cells^2, classes + 5,)
			p = pred[i]

			# (cells^2,)
			p_probs = p[..., self.classes]

			# (cells^2,)
			indices = torch.argsort(p_probs, descending=True)

			# (cells^2, classes + 5,) - sorted
			pred[i] = p[indices]
	
		# allocate a container from the stats
		iou_c_stats = {}
		for i in self.iou_thresholds:
			iou_c_stats[i] = {}
			for j in range(self.classes):
				iou_c_stats[i][j] = {}

		# calculate the precision vs recall graph for all thresholds and for all classes
		for iou_threshold in self.iou_thresholds:
			for c in range(self.classes):
				# combine the stats from all the batches
				num_labels = 0
				stats = []
				for item in self.calculate_image_stats(pred, label, iou_threshold, c):
					num_labels += item['num_labels']
					stats += item['stats']
				
				# if there are no positive labels, ignore this case
				if num_labels == 0:
					iou_c_stats[iou_threshold][c] = {
						'p': [],
						'r': [],
						't': [],
					}
				else:
					# sort by box confidence
					stats.sort(key=lambda s: s[1], reverse=True)

					# calculate precision, recall, confidence
					precision = []
					recall = []
					confidence = []
					tps = 0
					positive_predictions = 0
					for box_m, box_c in stats:
						if box_m == 'TP':
							tps += 1
						
						positive_predictions += 1
						precision.append(tps / positive_predictions)
						recall.append(tps / num_labels)
						confidence.append(box_c)
					
					iou_c_stats[iou_threshold][c] = {
						'p': precision,
						'r': recall,
						't': confidence,
					}
		
		return iou_c_stats

	@classmethod
	def to_absolute(cls, boxes):
		'''
		input:
			boxes: (batch, cells, cells, classes + 5)
		output:
			absolute boxes: (batch, cells^2, classes + 5)
		'''
		# extract the dimensions
		batch, cells_a, cells_b, cp5 = boxes.shape
		classes = cp5 - 5
		
		# absolute cell size (width, height)
		cell_size = torch.Tensor([1 / cells_b, 1 / cells_a])

		# calculate the offset of each cell
		# such that offset[i][j] = [cell's column, cell's row]
		# (cells, cells, 2)
		offsets = torch.Tensor([
			list(zip(range(cells_b), [i] * cells_b))
			for i in range(cells_a)
		])

		# expand to match the size of boxes
		# (1, cells, cells, offset)
		offsets = offsets.unsqueeze(0)

		# multiply the offsets by the cell sizes to get the absolute cell locations
		cell_location = offsets * cell_size

		# un-normalize the objects's locations
		# location <- cell location + cell size * location
		boxes[..., classes + 1: classes + 3] *= cell_size
		boxes[..., classes + 1: classes + 3] += cell_location

		# un-normalize the objects' sizes
		boxes[..., classes + 3: classes + 5] *= cell_size

		# flatten
		# (batch, cells^2, classes + 5)
		return boxes.reshape((batch, cells_a * cells_b, cp5))

	@classmethod
	def iou_vec(cls, target, boxes):
		'''
		Calculates the IOU of each box in `boxes` with the target.
		input:
			target: (classes + 5,)
			boxes: (x, classes + 5)
		output: the iou
			(x,)
		'''
		# extract the number of classes
		classes = target.shape[0] - 5

		# match the dimensions
		# (1, classes + 5)
		target = target.unsqueeze(0)
		
		# extract the target's location and size
		# (1, 2)
		t_l = target[..., classes + 1: classes + 3]
		t_s = target[..., classes + 3: classes + 5]

		# extract the boxes' locations and sizes
		# (x, 2)
		b_l = boxes[..., classes + 1: classes + 3]
		b_s = boxes[..., classes + 3: classes + 5]

		# extract the target's points - top left and bottom right
		# (1, 2)
		t_tl = t_l - t_s / 2
		t_br = t_l + t_s / 2

		# extract the boxes's points - top left and bottom right
		# (x, 2)
		b_tl = b_l - b_s / 2
		b_br = b_l + b_s / 2

		# x segment shared between the boxes
		# (x,)
		delta_x = (
			torch.minimum(t_br[..., 0], b_br[..., 0]) -
			torch.maximum(t_tl[..., 0], b_tl[..., 0])
		).clamp(0.0, None)

		# y segment shared between the boxes
		# (x,)
		delta_y = (
				torch.minimum(t_br[..., 1], b_br[..., 1]) -
				torch.maximum(t_tl[..., 1], b_tl[..., 1])
		).clamp(0.0, None)

		# intersection and union
		# (x,)
		intersection = delta_x * delta_y
		union = b_s[..., 0] * b_s[..., 1] + t_s[..., 0] * t_s[..., 1] - intersection

		# iou with numerical stability
		return intersection / (union + 1e-5)

	@classmethod
	def iou_mat(cls, a, b):
		'''
		input: matrices with boxes
			a: (x, classes + 5)
			b: (y, classes + 5)
		output: a matrix such that mat[i][j] = iou(a[i], b[j])
			mat: (x, y)
		'''
		# list of (y,), len = x
		rows = [Stats.iou_vec(a[i], b) for i in range(a.shape[0])]
		
		# reshape to list of (1, y), and concat to (x, y)
		return torch.cat([row.unsqueeze(0) for row in rows], dim=0)

	@ classmethod
	def filter_class(cls, t, c):
		'''
		input:
			t: (?, classes + 5)
			c: int
		out:
			filtered: (?, classes + 5) such that argmax(filtered[:classes]) = c
		'''
		classes = t.shape[1] - 5
		
		# extract the classes
		t_classes = t[:, :classes]

		# arg-max over the classes
		idx = torch.argmax(t_classes, dim=-1)
		
		# filter by the arg-max
		filtered = idx == c
		return t[filtered]
	
	def calculate_image_stats(self, pred, label, iou_threshold, c):
		'''
		input:
			pred: list of (?, classes + 5) - predicted boxes
			label: list of (?, classes + 5) - label boxes
			iou_threshold - predictions share an iou above this threshold count as
				true positives
			c - the class
		output:
			image_stats: list of stats for each item in the batch.
			stats:
				num_labels - number of labels
				stats - list of tuples: ('TP' or 'FP', the confidence of that box)
		'''
		image_stats = []

		for b in range(self.batches):
			# (x, 5)
			p = Stats.filter_class(pred[b], c)
			
			# (y, 5)
			l = Stats.filter_class(label[b], c)

			stats = []

			# if there are no positive labels then all boxes are false positives
			if l.shape[0] == 0:
				for i in range(p.shape[0]):
					confidence = p[i][self.classes].item()
					classification = 'FP'
					stats.append((classification, confidence))
			elif p.shape[0] == 0:
				# there are no predictions, ignore
				pass
			else:
				# (x, y)
				ious = Stats.iou_mat(p, l)

				for i in range(p.shape[0]):
					# (y,)
					label_ious = ious[i]

					# find the label with the maximum IOU
					# (1,)
					val, idx = torch.max(label_ious, dim=0)

					# if the IOU is above the threshold, mark the box as a true positive,
					# and consume the label
					confidence = p[i][self.classes].item()
					if val > iou_threshold:
						classification = 'TP'
						ious[: idx] = -1
					else:
						classification = 'FP'
					
					stats.append((classification, confidence))

			image_stats.append({
				'num_labels': l.shape[0],
				'stats': stats,
			})
		
		return image_stats

	def calculate_map(self):
		def area(xs, ys):
			assert len(xs) == len(ys)
			a = 0
			for i in range(len(xs) - 1):
				delta_x = xs[i + 1] - xs[i]
				y2 = ys[i + 1]
				y1 = ys[i]
				a += delta_x / 2 * (3 * y2 - y1)
			return a
		
		iou_c_stats = self.iou_c_stats()

		# calculate the mean average precision
		map_xs = []
		map_ys = []
		for iou_threshold in self.iou_thresholds:
			map_xs.append(iou_threshold)
			map_ys.append(sum(
				area(
					iou_c_stats[iou_threshold][c]['r'],
					iou_c_stats[iou_threshold][c]['p']
				)
				for c in range(self.classes)
			) / self.classes)
		
		return map_xs, map_ys
