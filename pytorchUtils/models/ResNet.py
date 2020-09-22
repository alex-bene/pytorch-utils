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

A slightly modified ResNet model class.
"""

import torch
from torchvision.models import resnet
from .BaseClassificationModel import BaseClassificationModel

class ResNet(resnet.ResNet, BaseClassificationModel):
	def __init__(self, pretrained=True, feature_extraction=True,
	             pretrained_path_or_url=None, use_cuda=True, num_classes=1000, resnet_type='resnet18', **kwargs):
		block, layers, kwargs = self.get_resnet_specs(resnet_type, kwargs)

		# to load the "default" ImageNet weigths the output classes must be 1000
		if pretrained and pretrained_path_or_url is None:
			num_classes_t = 1000
			self.pretrained_path_or_url = resnet.model_urls[resnet_type]
		else:
			num_classes_t = num_classes
			self.pretrained_path_or_url = pretrained_path_or_url

		super().__init__(block=block, layers=layers, num_classes=num_classes_t, **kwargs)

		BaseClassificationModel.__init__(self, pretrained_path_or_url=self.pretrained_path_or_url, use_cuda=use_cuda,
		                                 pretrained=pretrained, feature_extraction=feature_extraction)

		if pretrained and pretrained_path_or_url is None and num_classes != 1000:
			num_ftrs = self.fc.in_features
			self.fc = torch.nn.Linear(num_ftrs, num_classes)

	def get_resnet_specs(self, resnet_type, kwargs):
		if resnet_type not in resnet.model_urls.keys():
			raise ValueError(f"resnet_type must be one of {list(resnet.model_urls.keys())}.")

		if resnet_type=='resnet18' or resnet_type=='resnet34':
			block  = resnet.BasicBlock
		else:
			block  = resnet.Bottleneck

		if resnet_type=='resnet18':
			layers = [2, 2, 2, 2]
		elif '34' in resnet_type or '50' in resnet_type:
			layers = [3, 4, 6, 3]
		elif '101' in resnet_type:
			layers = [3, 4, 23, 3]
		elif resnet_type=='resnet152':
			layers = [3, 8, 36, 3]

		if resnet_type=='resnext50_32x4d':
			kwargs['width_per_group'] = 4
		elif resnet_type=='resnext101_32x8d':
			kwargs['width_per_group'] = 8
		elif resnet_type=='wide_resnet50_2':
			kwargs['width_per_group'] = 64 * 2
		elif resnet_type=='wide_resnet101_2':
			kwargs['width_per_group'] = 64 * 2

		if resnet_type.startswith('resnext'):
			kwargs['groups'] = 32

		return block, layers, kwargs