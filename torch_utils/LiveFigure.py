"""
LiveFigure.py

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

A class to create real time figures for the training process of a neural network.
"""

import numpy as np
from matplotlib import pyplot as plt
from IPython.display import clear_output, display

class LiveFigure():
	def __init__(self, fig=None, figsize=None, sleep=True):
		if fig is None:
			self.fig, self.axes = plt.subplots(1, 1, figsize=figsize)
			self.axes = [self.axes]
		else:
			self.fig = fig
			self.axes = self.fig.axes
		if not isinstance(self.fig.axes, list):
			self.axes  = [self.fig.axes]

		self.sleep = sleep

	def update(self, data, axes=None):
		if axes is None:
			axes = np.arange(len(self.axes))

		for (x, y, label, title), ax_idx in zip(data, axes):
			self.axes[ax_idx].cla()
			self.axes[ax_idx].plot(np.transpose(x), np.transpose(y))
			self.axes[ax_idx].set_title(title)
			self.axes[ax_idx].legend(label)
		clear_output(wait=True)
		display(self.fig)
		if self.sleep:
			plt.pause(0.5)