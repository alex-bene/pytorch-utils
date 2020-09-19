"""
ResNetMNIST.py

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

A modified ResNet18 model to support the 10-class 1 channel samples of the MNIST dataset
"""

import torch
import torchvision
from torchvision.models import resnet

class ResNet18MNIST(resnet.ResNet):
	def __init__(self, pretrained=True, input_size=224, feature_extraction=True, load_state_path=None):
		super().__init__(block=resnet.BasicBlock, layers=[2, 2, 2, 2])

		resnet18_path = 'https://download.pytorch.org/models/resnet18-5c106cde.pth'

		self.input_size = input_size

		if pretrained:
			state_dict = torch.utils.model_zoo.load_url(resnet18_path)
			self.load_state_dict(state_dict)
			# pretrained model have 224x224 input
			if self.input_size != 224:
				print("Input size is set to 224 because of the use of pretrained weights")
				self.input_size = 224

		if feature_extraction:
			for param in self.parameters():
				param.requires_grad = False

		# change output layer to fit MNIST (10 classes)
		num_ftrs = self.fc.in_features
		self.fc  = torch.nn.Linear(num_ftrs, 10)

		self.load_state(load_state_path)

		if torch.cuda.is_available():
			self.cuda()

	def load_state(self, load_state_path=None):
		if load_state_path is not None:
			self.load_state_dict(torch.load(load_state_path))

	def freeze_layers(self, layers):
		# all until that layer
		if isinstance(layers, basestring ):
			found_name = False
			for name, params in self.named_parameters():
				if name == layers:
					found_name = True
				params.requires_grad = found_name
		# the first {layers} layers
		elif isinstance(layers, int):
			nop = len(self.named_parameters()) # number of parameters
			found_layer = False
			for cnt, params in enumerate(self.parameters()):
				if (nop - cnt) == layers:
					found_name = True
				params.requires_grad = found_name
		# only those with matching names
		elif isinstance(layers, list) and all(isinstance(layer, basestring) for layer in layers):
			for name, params in self.named_parameters():
				params.requires_grad = (name in layers)
		# only those with matching names
		elif isinstance(layers, list) and all(isinstance(layer, int) for layer in layers):
			for cnt, params in enumerate(self.parameters()):
				params.requires_grad = ((nop - cnt) in layers)
		else:
			raise TypeError("The argument must be a model parameter name or "
		                    "a list of them or an integer or a list of them")

	def print_trainable_params(self):
		params_to_update = model.parameters()
		print("Parameters to train:")
		cnt = 0
		for name, param in self.model.named_parameters():
			if param.requires_grad == True:
				print("\t", name)
				cnt += 1
		print(f"There are {cnt} trainable parameters")
		