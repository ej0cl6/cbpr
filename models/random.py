import numpy as np

class Random:
	def __init__(self, K, D):
		self.K = K
		self.D = D

	def get_action(self, x):
		return np.random.randint(self.K)

	def update(self, px, pa, pr):
		return
