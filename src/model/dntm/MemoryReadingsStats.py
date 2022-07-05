import torch
import os
from glob import glob
import matplotlib.pyplot as plt
import seaborn as sns


class MemoryReadingsStats:
	def __init__(self, path=None):
		self.path = path + '/'
		self.memory_readings = None  # will have shape (dataset_size, memory_size)
		self.readings_variance = None
		self.kl_divergence = None
		self.random_projections = None
		self.random_matrix = None


	def load_memory_readings(self, epoch):
		if (self.path is not None) and (self.memory_readings is None):
			self.memory_readings = torch.concat([torch.load(path) for path in glob(self.path + 'memory_readings' + f"*_epoch{epoch}.pt")], dim=1)
			self.memory_readings = self.memory_readings.detach().cpu()


	def update_memory_readings(self, batch_readings, epoch=0):
		batch_readings = batch_readings.detach().cpu()
		if self.path is None:
			if self.memory_readings is None:
				self.memory_readings = batch_readings
			else:
				self.memory_readings = torch.concat((self.memory_readings, batch_readings))
		else:
			num_saved_readings = len(glob(self.path + 'memory_readings' + f"*_epoch{epoch}.pt"))
			torch.save(batch_readings, self.path + "memory_readings_{0:03}".format(num_saved_readings + 1) + f"_epoch{epoch}.pt")


	def reset(self):
		self.memory_readings = None
		self.readings_variance = None
		self.kl_divergence = None
		self.random_projections = None


	def compute_readings_variance(self):
		assert self.memory_readings is not None
		self.readings_variance = torch.var(self.memory_readings, dim=(0, 1), unbiased=False)
		return self.readings_variance


	def compute_readings_kl_divergence(self):
		assert self.memory_readings is not None
		kl_div = torch.nn.functional.kl_div
		sample = torch.rand(self.memory_readings.shape)
		self.kl_divergence = kl_div(self.memory_readings, sample)
		return self.kl_divergence


	def init_random_matrix(self, memory_size):
		if self.random_matrix is None:
			self.random_matrix = torch.rand((memory_size, 2))


	def compute_random_projections(self):
		assert self.random_matrix is not None
		assert self.memory_readings is not None
		self.random_projections = self.memory_readings.T @ self.random_matrix
		return self.random_projections


	def compute_stats(self):
		assert self.memory_readings is not None
		self.compute_readings_variance()
		self.compute_readings_kl_divergence()
		self.compute_random_projections()


	def get_stats(self):
		var = f"Readings variance: {self.readings_variance}\n"
		kl = f"Readings KL divergence from uniform distribution: {self.kl_divergence}\n"
		return var + kl


	def plot_random_projections(self):
		assert self.random_projections is not None
		fig, ax = plt.subplots(1, 1, figsize=(8,8))
		_ = sns.jointplot(x=self.random_projections[:, 0],
                  		  y=self.random_projections[:, 1], ax=ax)
		num_saved_projection_plots = len(glob(self.path + 'memory_readings_projections_*'))
		plt.savefig(self.path + "memory_readings_projections_epoch{0:03}.png".format(num_saved_projection_plots))


	def __repr__(self):
		return self.get_stats()
