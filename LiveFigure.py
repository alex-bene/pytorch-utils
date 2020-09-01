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