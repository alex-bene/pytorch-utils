import time
import copy
import torch
import numpy as np
from .LiveFigure import LiveFigure
from matplotlib import pyplot as plt

# progress bar (works in Jupyter notebook too!)
# try:
# 	get_ipython
# 	import tqdm.notebook as tqdm
# except:
# 	import tqdm
from tqdm.autonotebook import trange, tqdm

class Pipeline():
	def __init__(self, model, device, optimizer, criterion,
	                   trainloader, testloader, valloader=None,
	                   train_losses=None, val_losses=None, train_accuracies=None, val_accuracies=None,
	                   best_model=None, live_plot=False, time_elapsed=0.0, init_epoch=1):
		self.model        = model
		self.device       = device
		self.optimizer    = optimizer
		self.criterion    = criterion
		self.trainloader  = trainloader
		self.valloader    = valloader
		self.testloader   = testloader
		self.init_epoch   = init_epoch
		self.time_elapsed = time_elapsed

		self.set_live_plot(   live_plot   )

		self.update_accuracies_history(train_accuracies, val_accuracies)
		self.update_losses_history(    train_losses    , val_losses    )
		self.update_best_model(        best_model                      )

	def set_model(self, model):
		self.model = model

	def set_current_epoch(self, set_current_epoch):
		self.init_epoch = set_current_epoch

	def set_dataloaders(self, trainloader=None, testloader=None, valloader=None):
		if trainloader is not None:
			self.trainloader = trainloader
		if valloader is not None:
			self.valloader   = valloader
		if testloader is not None:
			self.testloader  = testloader

	def set_live_plot(self, live_plot=False):
		if live_plot:
			self.live_figure = LiveFigure(plt.subplots(2, 2, figsize=(30, 12))[0], sleep=False)

		self.live_plot = live_plot

	def update_losses_history(self, train_losses=None, val_losses=None):
		if train_losses is None:
			train_losses = []
		if val_losses is None:
			val_losses   = []

		self.train_losses = train_losses
		self.val_losses   = val_losses

	def update_accuracies_history(self, train_accuracies=None, val_accuracies=None):
		if train_accuracies is None:
			train_accuracies = []
		if val_accuracies is None:
			val_accuracies   = []
			self.best_val_acc = 0.0
		else :
			self.best_val_acc = max(val_accuracies)

		self.train_accuracies = train_accuracies
		self.val_accuracies   = val_accuracies

	def update_best_model(self, best_model=None):
		if best_model is None:
			best_model = copy.deepcopy(self.model)#.state_dict())

		self.best_model = best_model

	def moving_average(self, a, n=3):
		n  += 1 - n%2 # odd numbers to be able to center each data value in the window
		pad = (n-1)//2
		ret = np.pad(a, (pad + 1, pad), 'edge')
		ret = np.cumsum(ret, dtype=float)
		ret = ret[n:] - ret[:-n]

		return ret / n

	def update_live_plot(self):
		self.live_figure.update([
								[[np.arange(len(self.train_losses)),     np.arange(len(self.train_losses    ))],
								[self.train_losses,     self.moving_average(self.train_losses    , 3)],
								['raw', '3-moving average'], 'Traning Loss'       ], # train loss
								[[np.arange(len(self.train_accuracies)), np.arange(len(self.train_accuracies))],
								[self.train_accuracies, self.moving_average(self.train_accuracies, 3)],
								['raw', '3-moving average'], 'Traning Accuracy'   ], # train accuracy
								[[np.arange(len(self.val_losses)),       np.arange(len(self.val_losses      ))],
								[self.val_losses,       self.moving_average(self.val_losses      , 3)],
								['raw', '3-moving average'], 'Validation Loss'    ], # validation loss
								[[np.arange(len(self.val_accuracies)),   np.arange(len(self.val_accuracies  ))],
								[self.val_accuracies,   self.moving_average(self.val_accuracies  , 3)],
								['raw', '3-moving average'], 'Validation Accuracy'] # validation accuracy
								])

	def train_one_epoch(self):
		# ----------------- TRAINING ----------------- #
		samples        = 0
		train_loss     = 0.0
		train_accuracy = 0.0
		progress       = tqdm(enumerate(self.trainloader), desc="TRAINING | Loss: - Accuracy:", total=len(self.trainloader), unit='batches', ncols='100%')

		self.model.train()
		for cnt, (inputs, labels) in progress:
			inputs = inputs.to(self.device)
			labels = labels.to(self.device)

			# zero the parameter gradients
			self.optimizer.zero_grad()

			# forward + backward + optimize
			outputs = self.model(inputs)
			loss    = self.criterion(outputs, labels)

			loss.backward()
			self.optimizer.step()

			train_loss     += loss
			_, predicted    = torch.max(outputs.data, 1)
			train_accuracy += (predicted == labels).sum().item()

			samples += len(outputs)
	        # updating progress bar
			progress.set_description(f"TRAINING | Loss: {train_loss/(cnt+1):.4f} - Accuracy: {100*train_accuracy/samples:.2f}")

		# calculate statistics
		train_loss     /= len(self.trainloader)
		train_accuracy *= 100/samples

		# ----------------- VALIDATION ----------------- #
		if self.valloader is not None:
			samples      = 0
			val_loss     = 0.0
			val_accuracy = 0.0
			progress     = tqdm(enumerate(self.valloader), desc="VALIDATION | Loss: - Accuracy:", total=len(self.valloader), unit='batches', ncols='100%')

			self.model.eval()
			for cnt, (inputs, labels) in progress:
				inputs = inputs.to(self.device)
				labels = labels.to(self.device)

				with torch.no_grad():
					outputs = self.model(inputs)
					loss    = self.criterion(outputs, labels)

					val_loss     += loss
					_, predicted  = torch.max(outputs.data, 1)
					val_accuracy += (predicted == labels).sum().item()

				samples += len(outputs)
				# updating progress bar
				progress.set_description(f"VALIDATION | Loss: {val_loss/(cnt+1):.4f} - Accuracy: {100*val_accuracy/samples:.2f}")

			# calculate statistics
			val_loss     /= len(self.valloader)
			val_accuracy *= 100/samples

		else:
			val_loss     = None
			val_accuracy = None

		return train_loss, val_loss, train_accuracy, val_accuracy

	def training(self, epochs, live_plot_every=1, earlystopping=None):
		since = time.time()

		for self.epoch in trange(self.init_epoch, self.init_epoch + epochs): # loop over the dataset multiple times
			train_loss, val_loss, train_accuracy, val_accuracy = self.train_one_epoch()

			self.train_losses.append(    train_loss    )
			self.train_accuracies.append(train_accuracy)

			if self.valloader is not None:
				if val_accuracy >= self.best_val_acc:
					self.best_val_acc = val_accuracy
					self.best_model   = copy.deepcopy(self.model)#.state_dict())

				self.val_losses.append(    val_loss    )
				self.val_accuracies.append(val_accuracy)

			if self.live_plot and (not (self.epoch)%live_plot_every):
				self.update_live_plot()

			if earlystopping is not None:
				earlystopping(val_loss, model)
				if earlystopping.early_stop:
					print("Early stopping")
					break

		plt.pause(1)
		self.init_epoch = self.epoch + 1

		# calculate statistics
		if self.live_plot:
			self.update_live_plot()

		self.time_elapsed = time.time() - since
		print(f'Training complete in {self.time_elapsed // 60:.0f}m {self.time_elapsed % 60:.0f}s')
		if self.valloader is not None:
			print(f"{self.epoch} epochs done. Best validation accuracy is {max(self.val_accuracies):0.3f}. Best validation loss is {min(self.val_losses):0.3f}")

		return self.train_losses, self.val_losses, \
		       self.train_accuracies, self.val_accuracies, \
			   self.best_model


	def test(self, model=None, testloader=None):
		if model      is None:
			model      = self.model
		if testloader is None:
			testloader = self.testloader

		samples  = 0
		avg_loss = 0.0
		accuracy = 0.0
		progress = tqdm(enumerate(testloader), desc="Loss: - Accuracy:", total=len(testloader), unit='batches', ncols='100%')

		model.eval()
		for cnt, (inputs, labels) in progress:
			inputs = inputs.to(self.device)
			labels = labels.to(self.device)

			with torch.no_grad():
				outputs = model(inputs)
				loss    = self.criterion(outputs, labels)

				avg_loss     += loss
				_, predicted  = torch.max(outputs.data, 1)
				accuracy += (predicted == labels).sum().item()

			samples += len(outputs)
	        # updating progress bar
			progress.set_description(f"Loss: {avg_loss/(cnt+1):.4f} - Accuracy: {100*accuracy/samples:.2f}")

		# calculate statistics
		avg_loss /= len(testloader)
		accuracy *= 100/samples

		return avg_loss.item(), accuracy