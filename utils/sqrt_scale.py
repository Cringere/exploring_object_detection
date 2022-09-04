import torch
import torch.nn as nn

class SqrtScale(nn.Module):
	'''
	f(x) = sqrt(|x|) * sign(x)
	'''
	def __init__(self):
		super().__init__()
	
	def forward(self, x):
		return torch.sign(x) * torch.sqrt(torch.abs(x))
