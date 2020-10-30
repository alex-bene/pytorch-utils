"""
BaseClassificationModel.py

MIT License

Copyright (c) 2020 Alexandros Benetatos

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

"""
DESCRIPTION

A BaseClassificationModel enhances with some functions for use ONLY as the last parent class to a neural network class.
"""

import torch
from .pretrained.Pretrained import PretrainedModelUrl

class BaseClassificationModel():
	def __init__(self, pretrained=True, pretrained_path_or_url=None,
	             feature_extraction=True, use_cuda=True, num_classes=None):
		super().__init__()

		if pretrained:
			if isinstance(pretrained_path_or_url, PretrainedModelUrl) and pretrained_path_or_url.url.startswith('https:'):
				state_dict = torch.utils.model_zoo.load_url(pretrained_path_or_url.url)
			elif isinstance(pretrained_path_or_url, str) and pretrained_path_or_url.startswith('https:'):
				state_dict = torch.utils.model_zoo.load_url(pretrained_path_or_url)
			elif isinstance(pretrained_path_or_url, str):
				state_dict = torch.load(pretrained_path_or_url)
			else:
				raise ValueError("If pretrained is 'True' pretrained_path_or_url must be a string or a PretrainedModelUrl object")

			self.load_state_dict(state_dict)

		if feature_extraction:
			self.freeze_layers(-2) #fc.weight and fc.bias

		if num_classes is not None:
			num_ftrs = self.fc.in_features
			self.fc = torch.nn.Linear(num_ftrs, num_classes)

		if use_cuda:
			if torch.cuda.is_available():
				self.cuda()
			else:
				print("CUDA is not available at the moment --> using CPU")

	def load_state_from_path(self, load_state_path=None):
		if load_state_path is None:
			raise ValueError("load_state_path can not be 'None'")
		else:
			self.load_state_dict(torch.load(load_state_path))

	def freeze_layers(self, layers=None):
		# freeze all
		if layers is None:
			for param in self.parameters():
				param.requires_grad = False
		# all until that layer
		elif isinstance(layers, str):
			found_name = False
			for name, params in self.named_parameters():
				if name == layers:
					found_name = True
				params.requires_grad = found_name
		# all but the last {-layers} layers
		elif isinstance(layers, int) and layers < 0:
			nop = sum(1 for _ in self.parameters()) # number of layers
			found_layer = False
			for cnt, params in enumerate(self.parameters()):
				if (nop - cnt) == -layers:
					found_layer = True
				params.requires_grad = found_layer
		# the first {layers} layers
		elif isinstance(layers, int) and layers >= 0:
			found_layer = False
			for cnt, params in enumerate(self.parameters()):
				if cnt == layers:
					found_layer = True
				params.requires_grad = found_layer
		# only those with matching names
		elif isinstance(layers, list) and all(isinstance(layer, str) for layer in layers):
			for name, params in self.named_parameters():
				params.requires_grad = name not in layers
		# only those with matching 'layer number'
		elif isinstance(layers, list) and all(isinstance(layer, int) for layer in layers):
			for cnt, params in enumerate(self.parameters()):
				params.requires_grad = cnt not in layers
		else:
			raise TypeError("The argument must either be 'None' (freeze all layers), or a model "
			                "parameter name (or a list of them) or an integer (or a list of them)")

	def print_trainable_params(self):
		print("Parameters to train:")
		cnt = 0
		for name, param in self.named_parameters():
			if param.requires_grad == True:
				print("\t", name)
				cnt += 1
		print(f"There are {cnt} trainable parameters")