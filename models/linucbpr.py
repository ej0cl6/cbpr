import numpy as np

class LinUCBPR:
	def __init__(self, K, D, alpha=0.1):
		self.K = K
		self.D = D
		self.alpha = alpha
		self.vA = [np.eye(D) for k in xrange(K)]
		self.vb = [np.zeros((self.D,)) for k in xrange(K)]
		self.hA = [np.eye(D) for k in xrange(K)]
		self.hb = [np.zeros((self.D,)) for k in xrange(K)]

	def get_action(self, x):
		expected = np.array([np.dot(x, np.dot(self.hA[k], self.hb[k])) for k in xrange(self.K)])
		uncertainty = np.array([np.sqrt(np.dot(x, np.dot(self.hA[k], x))) for k in xrange(self.K)])
		scores = np.random.random(self.K)*1e-10 + expected + self.alpha*uncertainty
		
		a = scores.argmax()
		r = expected[a]
		Ax = np.dot(self.hA[a], x)
		self.hA[a] -= np.dot(Ax[:, None], Ax[None, :])/(1.0+np.dot(Ax, x))
		self.hb[a] += r*x
		return a

	def update(self, px, pa, pr):
		for x, a, r in zip(px, pa, pr):
			Ax = np.dot(self.vA[a], x)
			self.vA[a] -= np.dot(Ax[:, None], Ax[None, :])/(1.0+np.dot(Ax, x))
			self.vb[a] += r*x

		for k in xrange(self.K):
			self.hA[k][:] = self.vA[k]
			self.hb[k][:] = self.vb[k]
