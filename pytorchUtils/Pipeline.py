"""
Pipeline.py

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

A class to set a training and testing pipeline for neural networks with pytorch.
"""

import time
import copy
import torch
import numpy as np
from .LiveFigure import LiveFigure
from matplotlib import pyplot as plt
from tqdm.autonotebook import trange, tqdm

# progress bar (works in Jupyter notebook too!)
# try:
# 	get_ipython
# 	import tqdm.notebook as tqdm
# except:
# 	import tqdm

class Pipeline():
	def __init__(self, model, device, optimizer, criterion,
	                   trainloader, testloader, valloader=None, hide_sub_bars=True, hide_all_bars=False,
	                   train_losses=None, val_losses=None, train_accuracies=None, val_accuracies=None,
	                   best_loss_model=None, best_acc_model=None, best_model_val_loss=None, best_model_val_acc=None,
					   live_plot=False, time_elapsed=0.0, init_epochs=0, use_accuracy=True):
		self.model         = model
		self.device        = device
		self.optimizer     = optimizer
		self.criterion     = criterion
		self.trainloader   = trainloader
		self.valloader     = valloader
		self.testloader    = testloader
		self.epochs        = init_epochs
		self.time_elapsed  = time_elapsed
		self.hide_sub_bars = hide_sub_bars or hide_all_bars
		self.hide_top_bars = hide_all_bars
		self.use_accuracy  = use_accuracy

		self.set_live_plot(live_plot)

		self.update_losses_history(    train_losses   , val_losses         )
		self.update_best_loss_model(   best_loss_model, best_model_val_loss)

		if use_accuracy:
			self.update_accuracies_history(train_accuracies, val_accuracies     )
			self.update_best_acc_model(    best_acc_model  , best_model_val_acc )

	def set_model(self, model):
		self.model = model

	def set_current_epoch(self, set_current_epoch):
		self.epochs = set_current_epoch

	def set_dataloaders(self, trainloader=None, testloader=None, valloader=None):
		if trainloader is not None:
			self.trainloader = trainloader
		if valloader is not None:
			self.valloader   = valloader
		if testloader is not None:
			self.testloader  = testloader

	def set_live_plot(self, live_plot=False):
		if live_plot:
			cols = 2 if self.valloader is not None else 1
			rows = 2 if self.use_accuracy else 1
			self.live_figure = LiveFigure(plt.subplots(rows, cols, figsize=(15*rows, 6*cols))[0], sleep=False)

		self.live_plot = live_plot

	def update_losses_history(self, train_losses=None, val_losses=None):
		if train_losses is None:
			train_losses = []
		if val_losses is None:
			val_losses   = []
			self.best_val_loss = np.inf
		else:
			self.best_val_loss = min(val_losses)

		self.train_losses = train_losses
		self.val_losses   = val_losses

	def update_accuracies_history(self, train_accuracies=None, val_accuracies=None):
		if train_accuracies is None:
			train_accuracies = []
		if val_accuracies is None:
			val_accuracies   = []
			self.best_val_acc = 0.0
		else:
			self.best_val_acc = max(val_accuracies)

		self.train_accuracies = train_accuracies
		self.val_accuracies   = val_accuracies

	def update_best_acc_model(self, best_acc_model=None, best_model_val_acc=None):
		if best_acc_model is None:
			best_acc_model = copy.deepcopy(self.model) #.state_dict())
		if best_model_val_acc is None:
			best_model_val_acc = self.test(best_acc_model)
			if not isinstance(best_model_val_acc, float):
				best_model_val_acc = best_model_val_acc[1]

		self.best_acc_model = best_acc_model
		self.best_val_acc   = max(self.best_val_acc, best_model_val_acc)

	def update_best_loss_model(self, best_loss_model=None, best_model_val_loss=None):
		if best_loss_model is None:
			best_loss_model = copy.deepcopy(self.model) #.state_dict())
		if best_model_val_loss is None:
			best_model_val_loss = self.test(best_loss_model)
			if not isinstance(best_model_val_loss, float):
				best_model_val_loss = best_model_val_loss[0]

		self.best_loss_model = best_loss_model
		self.best_val_loss   = min(self.best_val_loss, best_model_val_loss)

	def moving_average(self, a, n=3):
		n  += 1 - n%2 # odd numbers to be able to center each data value in the window
		pad = (n - 1)//2
		ret = np.pad(a, (pad + 1, pad), 'edge')
		ret = np.cumsum(ret, dtype=float)
		ret = ret[n:] - ret[:-n]

		return ret / n

	def update_live_plot(self):
		update_lists = [
		            [[np.arange(1, 1+len(self.train_losses)),     np.arange(1, 1+len(self.train_losses    ))],
		             [self.train_losses,     self.moving_average(self.train_losses    , 3)],
		             ['raw', '3-moving average'], 'Traning Loss'       ] # train loss
		]
		if self.use_accuracy:
			update_lists.append(
			        [[np.arange(1, 1+len(self.train_accuracies)), np.arange(1, 1+len(self.train_accuracies))],
			         [self.train_accuracies, self.moving_average(self.train_accuracies, 3)],
			         ['raw', '3-moving average'], 'Traning Accuracy'   ] # train accuracy
			)
		if self.valloader is not None:
			update_lists.append(
			        [[np.arange(1, 1+len(self.val_losses)),       np.arange(1, 1+len(self.val_losses      ))],
			         [self.val_losses,       self.moving_average(self.val_losses      , 3)],
			         ['raw', '3-moving average'], 'Validation Loss'    ] # validation loss
			)
			if self.use_accuracy:
				update_lists.append(
			        [[np.arange(1, 1+len(self.val_accuracies)),   np.arange(1, 1+len(self.val_accuracies  ))],
			         [self.val_accuracies,   self.moving_average(self.val_accuracies  , 3)],
			         ['raw', '3-moving average'], 'Validation Accuracy'] # validation accuracy
				)

		self.live_figure.update(update_lists)

	def train_one_epoch(self):
		# ----------------- TRAINING ----------------- #
		samples    = 0
		train_loss = 0.0
		pbar_desc  = "TRAINING | Loss:"
		if self.use_accuracy:
			train_accuracy = 0.0
			pbar_desc      = "TRAINING | Loss: - Accuracy:"

		progress = tqdm(enumerate(self.trainloader), desc=pbar_desc, total=len(self.trainloader),
		                unit='batches', ncols='100%', disable=self.hide_sub_bars)

		self.model.train()
		for cnt, (inputs, labels) in progress:
			inputs = inputs.to(self.device)
			labels = labels.to(self.device)

			# zero the parameter gradients
			self.optimizer.zero_grad()

			# forward + backward + optimize
			outputs = self.model(inputs)
			loss    = self.criterion(outputs, labels)

			samples += len(outputs)

			loss.backward()
			self.optimizer.step()

			train_loss     += loss
			if self.use_accuracy:
				_, predicted    = torch.max(outputs.data, 1)
				train_accuracy += (predicted == labels).sum().item()

				# updating progress bar
				pbar_desc = f"TRAINING | Loss: {train_loss/(cnt+1):.4f} - Accuracy: {100*train_accuracy/samples:.2f}"
			else:
				# updating progress bar
				pbar_desc = f"TRAINING | Loss: {train_loss/(cnt+1):.4f}"
			progress.set_description(pbar_desc)

		# calculate statistics
		train_loss     /= len(self.trainloader)
		if self.use_accuracy:
			train_accuracy *= 100/samples

		# ----------------- VALIDATION ----------------- #
		if self.valloader is not None:
			samples      = 0
			val_loss     = 0.0
			pbar_desc  = "VALIDATION | Loss:"
			if self.use_accuracy:
				val_accuracy = 0.0
				pbar_desc    = "VALIDATION | Loss: - Accuracy:"

			progress = tqdm(enumerate(self.valloader), desc=pbar_desc, total=len(self.valloader),
							unit='batches', ncols='100%', disable=self.hide_sub_bars)

			self.model.eval()
			for cnt, (inputs, labels) in progress:
				inputs = inputs.to(self.device)
				labels = labels.to(self.device)

				with torch.no_grad():
					outputs = self.model(inputs)
					loss    = self.criterion(outputs, labels)

					val_loss += loss
					if self.use_accuracy:
						_, predicted  = torch.max(outputs.data, 1)
						val_accuracy += (predicted == labels).sum().item()

				samples += len(outputs)

				# updating progress bar
				pbar_desc = f"VALIDATION | Loss: {val_loss/(cnt+1):.4f}" + \
				            (f" - Accuracy: {100*val_accuracy/samples:.2f}" if self.use_accuracy else "")
				progress.set_description(pbar_desc)

			# calculate statistics
			val_loss /= len(self.valloader)
			if self.use_accuracy:
				val_accuracy *= 100/samples

		else:
			val_loss     = None
			if self.use_accuracy:
				val_accuracy = None

		if self.use_accuracy:
			return train_loss, val_loss, train_accuracy, val_accuracy
		else:
			return train_loss, val_loss

	def training(self, epochs, live_plot_every=1, earlystopping=None, show_training_stats=True):
		since       = time.time()
		init_epochs = self.epochs
		pbar_desc   = ("Validation" if self.valloader is not None else "Training") + " | Loss:"
		if self.use_accuracy:
			pbar_desc += " - Accuracy:"

		progress = tqdm(range(init_epochs, init_epochs + epochs), desc=pbar_desc,
		                unit='epochs', ncols='100%', disable=self.hide_top_bars)

		for self.epochs in progress: # loop over the dataset multiple times
			one_epoch = self.train_one_epoch()
			train_loss = one_epoch[0]
			val_loss   = one_epoch[1]

			self.train_losses.append(train_loss)
			if self.use_accuracy:
				train_accuracy = one_epoch[2]
				val_accuracy   = one_epoch[3]
				self.train_accuracies.append(train_accuracy)

			if self.valloader is not None:
				if val_loss <= self.best_val_loss:
					self.best_val_loss   = val_loss
					self.best_loss_model = copy.deepcopy(self.model) #.state_dict())
				self.val_losses.append(    val_loss    )

				if self.use_accuracy:
					if val_accuracy >= self.best_val_acc:
						self.best_val_acc    = val_accuracy
						self.best_acc_model  = copy.deepcopy(self.model) #.state_dict())
					self.val_accuracies.append(val_accuracy)

			if self.live_plot and (not (self.epochs + 1)%live_plot_every):
				self.update_live_plot()

			if earlystopping is not None:
				earlystopping(val_loss, model)
				if earlystopping.early_stop:
					print("Early stopping")
					break

	    # updating progress bar
		if self.valloader is not None:
			pbar_desc = f"Validation | Loss: {val_loss:.4f}" + \
			            (f" - Accuracy: {val_accuracy:.2f}" if self.use_accuracy else "")
		else:
			pbar_desc = f"Training | Loss: {train_loss:.4f}" + \
			            (f" - Accuracy: {train_accuracy:.2f}" if self.use_accuracy else "")

		progress.set_description(pbar_desc)

		plt.pause(1)
		self.epochs += 1

		# calculate statistics
		if self.live_plot:
			self.update_live_plot()

		self.time_elapsed = time.time() - since

		if show_training_stats:
			print(f'Training complete in {self.time_elapsed // 60:.0f}m {self.time_elapsed % 60:.0f}s')
			if self.valloader is not None:
				print(f"{self.epochs} epochs done. Best validation loss is {min(self.val_losses):0.4f}." + \
				      (f" Best validation accuracy is {max(self.val_accuracies):0.2f}." if self.use_accuracy else ""))

		if self.use_accuracy:
			return self.train_losses, self.val_losses, \
			       self.train_accuracies, self.val_accuracies, \
			       self.best_loss_model, self.best_acc_model
		else:
			return self.train_losses, self.val_losses, \
			       self.best_loss_model


	def test(self, model=None, testloader=None):
		if model      is None:
			model      = self.model
		if testloader is None:
			testloader = self.testloader

		samples   = 0
		avg_loss  = 0.0
		pbar_desc = "Loss:"
		if self.use_accuracy:
			accuracy  = 0.0
			pbar_desc = "Loss: - Accuracy:"

		progress = tqdm(enumerate(testloader), desc=pbar_desc, disable=self.hide_top_bars,
		                total=len(testloader), unit='batches', ncols='100%')

		model.eval()
		for cnt, (inputs, labels) in progress:
			inputs = inputs.to(self.device)
			labels = labels.to(self.device)

			with torch.no_grad():
				outputs = model(inputs)
				loss    = self.criterion(outputs, labels)

				avg_loss += loss
				if self.use_accuracy:
					_, predicted  = torch.max(outputs.data, 1)
					accuracy += (predicted == labels).sum().item()

			samples += len(outputs)

			# updating progress bar
			progress.set_description(f"Loss: {avg_loss/(cnt+1):.4f}" + 
				                     (f" - Accuracy: {100*accuracy/samples:.2f}" if self.use_accuracy else ""))

		# calculate statistics
		avg_loss /= len(testloader)
		if self.use_accuracy:
			accuracy *= 100/samples
			return avg_loss.item(), accuracy
		else:
			return avg_loss.item()

class ClassificationPipeline(Pipeline):
	def __init__(self, model, device, optimizer, criterion,
	                   trainloader, testloader, valloader=None, hide_sub_bars=True, hide_all_bars=False,
	                   train_losses=None, val_losses=None, train_accuracies=None, val_accuracies=None,
	                   best_loss_model=None, best_acc_model=None, best_model_val_loss=None, best_model_val_acc=None,
					   live_plot=False, time_elapsed=0.0, init_epochs=0):
	   super().__init__(model, device, optimizer, criterion,
	                   trainloader, testloader, valloader, hide_sub_bars, hide_all_bars,
	                   train_losses, val_losses, train_accuracies, val_accuracies,
	                   best_loss_model, best_acc_model, best_model_val_loss, best_model_val_acc,
					   live_plot, time_elapsed, init_epochs, True)

class RegressionPipeline(Pipeline):
	def __init__(self, model, device, optimizer, criterion,
	                   trainloader, testloader, valloader=None, hide_sub_bars=True, hide_all_bars=False,
	                   train_losses=None, val_losses=None, best_model=None, best_model_val_loss=None,
                       live_plot=False, time_elapsed=0.0, init_epochs=0):
	   super().__init__(model, device, optimizer, criterion,
	                   trainloader, testloader, valloader, hide_sub_bars, hide_all_bars,
	                   train_losses, val_losses, None, None,
	                   best_model, None, best_model_val_loss, None, live_plot, time_elapsed, init_epochs, False)