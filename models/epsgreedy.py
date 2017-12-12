import numpy as np

class EpsGreedy:
	def __init__(self, K, D, eps=0.1):
		self.K = K
		self.D = D
		self.eps = eps
		self.A = [np.eye(D) for k in xrange(K)]
		self.b = [np.zeros((self.D,)) for k in xrange(K)]

	def get_action(self, x):
		# random
		if np.random.random()<self.eps:
			return np.random.randint(self.K)
		# greedy
		scores = np.random.random(self.K)*1e-10 + [np.dot(x, np.dot(self.A[k], self.b[k])) for k in xrange(self.K)]
		return scores.argmax()

	def update(self, px, pa, pr):
		for x, a, r in zip(px, pa, pr):
			Ax = np.dot(self.A[a], x)
			self.A[a] -= np.dot(Ax[:, None], Ax[None, :])/(1.0+np.dot(Ax, x))
			self.b[a] += r*x
