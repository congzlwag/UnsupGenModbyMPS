# -*- coding: utf-8 -*-

import sys
sys.path.append('/home/qi/paper/AllNew')
# from shifter import BandS
from TrainingSetHonest import DataSet
from MPSserial import MPS
import numpy as np
from numpy.random import randint, rand
import os

# raw_data = BandS()
# np.save('BSdata.npy', raw_data)

def state2ind(state):
	return np.int(state.dot(2**np.arange(len(state))))

def remember(mps, steps, nsam):
	nsize = mps.space_size
	print('n_sample=%d'%nsam)
	current = randint(2, size=(nsam,nsize))

	for n in range(1,steps+1):
		if n%(steps//10) == 0: print('.',end='')
		flipper = randint(2, size=(nsam,nsize))
		new = current^flipper
		for x in range(nsam):
			prob  = mps.Give_probab(current[x])
			prob_ = mps.Give_probab(new[x])
			if prob_> prob or rand() < prob_/prob:
				current[x] = new[x]

	return current

def remember_zipper(mps, nsam):
	mps.left_cano()
	print('n_sample=%d'%nsam)
	sam = np.asarray([mps.Give_Sample() for _ in range(nsam)])
	return sam

def statistic(current):
	samprob = {}
	for x in current:
		xind = state2ind(x)
		if xind in samprob:
			samprob[xind] += 1
		else:
			samprob[xind] =1
	memory = [(x, samprob[x]) for x in samprob]
	memory = np.asarray(memory, dtype=[('x',int),('f',int)])
	return np.sort(memory, order='f')

# def samp_metro(mps, steps, nsam):
# 	nsize = mps.space_size
# 	print('n_sample=%d'%nsam)
# 	current = randint(2, size=(nsam,nsize))

# 	for n in range(1,steps+1):
# 		if n%(steps//10) == 0: print('.',end='')
# 		flipper = randint(2, size=(nsam,nsize))
# 		new = current^flipper
# 		for x in range(nsam):
# 			prob  = mps.Give_probab(current[x])
# 			prob_ = mps.Give_probab(new[x])
# 			if prob_> prob or rand() < prob_/prob:
# 				current[x] = new[x]
# 	return current


if __name__ == '__main__':
	# BSind = []
	# for s in raw_data:
	# 	BSind.append(state2ind(s))
	# np.save('BSind.npy', np.array(BSind))

	raw_data = np.load('BSdata.npy').reshape(-1,16)
	dataset  = DataSet(raw_data)
	os.mkdir('./0727-3')
	os.chdir('./0727-3')
	m = MPS(16)
	m.left_cano()
	m.normalize()

	m.cutoff = 5e-5
	m.descent_step_length = 0.05
	m.descent_steps = 10
	m.train(dataset, 2)

	m.saveMPS('BS')
	sam_zip = remember_zipper(m, 100000)
	np.save('sam_zip.npy', sam_zip)
	np.save('memo_zip.npy', statistic(sam_zip))
	# sam_met = remember(m, 5000, 10000)
	# np.save('sam_met.npy', sam_met)
	# np.save('memo_met.npy', statistic(sam_met))

	# m = MPS(16)
	# m.loadMPS('./0727-1/BSMPS_0105_27_Jul/')
	# m.left_cano()
	# np.save("samp_met.npy", samp_metro(m, 5000, 30))