import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from models.epsgreedy import EpsGreedy
from models.qpmd import QPMD
from models.linucb import LinUCB
from models.linucbpr import LinUCBPR
import sys

N = 5000 # number of rounds
D = 10 # dimension of context
K = 100 # number of actions
DELAY = 500 # size of pile

# set random seed
np.random.seed(1)

def gen_data(D, K, T, var):
	X = np.random.random((T, D))*2 - 1
	W = np.random.random((D, K))*2 - 1
	R = 1.0 + np.dot(X, W) + (np.random.random((T, K))*2 - 1)*var
	return X, R

x_data, r_data = gen_data(D, K, N, 0.05) # generate dataset

total_regret = np.zeros((N, 4)) # for recording

# run several experiments
sys.stderr.write('total  #####\n')
sys.stderr.write('runing ')
for i in xrange(5):
	# shuffle dataset
	idxs = np.arange(N)
	np.random.shuffle(idxs)
	x_data = x_data[idxs]
	r_data = r_data[idxs]

	# different models
	models = [EpsGreedy(K, D), QPMD(K, D), LinUCB(K, D), LinUCBPR(K, D)]

	# for recording rewards and actions
	rewards = np.zeros((N, len(models)))
	actions = np.zeros((N, len(models)), dtype=int)

	for rd in xrange(N):
		x = x_data[rd]
		r = r_data[rd]
		actions[rd, :] = [model.get_action(x) for model in models]
		rewards[rd, :] = r[actions[rd, :]]

		# update models
		if (rd+1)%DELAY==0:
			for i, model in enumerate(models):
				model.update(x_data[rd+1-DELAY:rd+1, :], actions[rd+1-DELAY:rd+1, i], rewards[rd+1-DELAY:rd+1, i])

	total_regret += r_data.max(axis=1)[:, None] - rewards
	sys.stderr.write('#')

sys.stderr.write('\nPlease see result.png\n')
avg_regret = total_regret/5
cul_regret = np.cumsum(avg_regret, axis=0)

# plot result
plt.figure()
show_x = np.arange(0, N, 250)
plt.plot(show_x, cul_regret[::250, 0], '-bd', fillstyle='none', linewidth=1.5, markeredgewidth=1.5, label='EpsGreedy')
plt.plot(show_x, cul_regret[::250, 1], '-c^', fillstyle='none', linewidth=1.5, markeredgewidth=1.5, label='QPM-D')
plt.plot(show_x, cul_regret[::250, 2], '-gs', fillstyle='none', linewidth=1.5, markeredgewidth=1.5, label='LinUCB')
plt.plot(show_x, cul_regret[::250, 3], '-ro', fillstyle='none', linewidth=1.5, markeredgewidth=1.5, label='LinUCBPR')
plt.xlabel('Round')
plt.ylabel('Regret')
plt.legend(loc='upper left')
plt.savefig('result.png')

