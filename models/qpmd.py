import numpy as np

class QPMD:
	def __init__(self, K, D, alpha=0.1):
		self.K = K
		self.D = D
		self.alpha = alpha
		self.A = [np.eye(D) for k in xrange(K)]
		self.b = [np.zeros((self.D,)) for k in xrange(K)]
		self.Q = [[] for k in xrange(K)]

	def get_action(self, x):
		expected = np.array([np.dot(x, np.dot(self.A[k], self.b[k])) for k in xrange(self.K)])
		uncertainty = np.array([np.sqrt(np.dot(x, np.dot(self.A[k], x))) for k in xrange(self.K)])
		scores = np.random.random(self.K)*1e-10 + expected + self.alpha*uncertainty
		a = scores.argmax()
		while len(self.Q[a])>0:
			x, r = self.Q[a][0]
			Ax = np.dot(self.A[a], x)
			self.A[a] -= np.dot(Ax[:, None], Ax[None, :])/(1.0+np.dot(Ax, x))
			self.b[a] += r*x
			self.Q[a] = self.Q[a][1:]
			expected[a] = np.dot(x, np.dot(self.A[a], self.b[a]))
			uncertainty[a] = np.sqrt(np.dot(x, np.dot(self.A[a], x)))
			scores = np.random.random(self.K)*1e-10 + expected + self.alpha*uncertainty
			a = scores.argmax()

		return a

	def update(self, px, pa, pr):
		for x, a, r in zip(px, pa, pr):
			self.Q[a].append((x, r))
			
