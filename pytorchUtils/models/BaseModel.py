"""
BaseModel.py

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

A BaseModel enhances with some functions for use ONLY as the last parent class to a neural network class.
"""

import torch

class BaseModel():
	def __init__(self, pretrained=True, pretrained_path=None,
	             feature_extraction=True, use_cuda=True):
		super().__init__()

		if pretrained:
			if pretrained_path is None:
				raise ValueError("If pretrained is 'True' pretrained_path can not be 'None'")
			elif pretrained_path.startswith('https:'):
				state_dict = torch.utils.model_zoo.load_url(pretrained_path)
			else:
				state_dict = torch.load(pretrained_path)
			self.load_state_dict(state_dict)

		if feature_extraction:
			self.freeze_layers(-1)

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
		elif isinstance(layers, basestring ):
			found_name = False
			for name, params in self.named_parameters():
				if name == layers:
					found_name = True
				params.requires_grad = found_name
		# all but the last {-layers} layers
		elif isinstance(layers, int) and layers < 0:
			nop = len(self.named_parameters()) # number of parameters
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
		elif isinstance(layers, list) and all(isinstance(layer, basestring) for layer in layers):
			for name, params in self.named_parameters():
				params.requires_grad = (name in layers)
		# only those with matching 'layer number'
		elif isinstance(layers, list) and all(isinstance(layer, int) for layer in layers):
			for cnt, params in enumerate(self.parameters()):
				params.requires_grad = ((nop - cnt) in layers)
		else:
			raise TypeError("The argument must either be 'None' (freeze all layers), or a model "
			                "parameter name (or a list of them) or an integer (or a list of them)")

	def print_trainable_params(self):
		params_to_update = model.parameters()
		print("Parameters to train:")
		cnt = 0
		for name, param in self.model.named_parameters():
			if param.requires_grad == True:
				print("\t", name)
				cnt += 1
		print(f"There are {cnt} trainable parameters")