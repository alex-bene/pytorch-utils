"""
PretrainedModels.py

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

A file to create the necessary structs for the pretrained models urls to be easily accessible.
"""

import os

class PretrainedModelUrls():
	def __init__(self):
		self.MNIST = MNISTpretrained()

class MNISTpretrained():
	def __init__(self):
		self.resnet18 = PretrainedModelUrl(os.path.join(os.getcwd(), 'MNIST__ResNet18Model__Adabound__25_epochs.pth'),
		                                   name = 'resnet18',
		                                   info = """optimizer: Adabound\n with lr=1e-3 and final_lr=0.1
		                                             criterion: CrossEntropyLoss
		                                             epochs   : 25
		                                             choice   : the model with the best validation accuracy
		                                             other    : transfer learning from ImageNet pretrained weights""")

class PretrainedModelUrl():
	def __init__(self, url, name, info):
		self.url = url
		self.name = name
		self.info = info

	def __call__(self):
		return self.url

	def __repr__(self):
		return self.url

url = PretrainedModelUrls()