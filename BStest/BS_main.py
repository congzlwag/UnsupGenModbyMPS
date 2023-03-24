# -*- coding: utf-8 -*-

import sys
sys.path.append('../')
from MPScumulant import MPS_c
import numpy as np
from numpy.random import randint, rand
import os


def state2ind(state):
	"""binary configuration -> int"""
	return np.int(state.dot(2**np.arange(len(state))))

def remember(mps, steps, nsam):
	"""Inference with Metropolis approach
	nsam: number of walkers = number of samples
	steps: number of steps they walk
	Their final states are returned
	"""
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
	"""Zipper sampling
	nsam: number of samples
	"""
	mps.left_cano()
	print('n_sample=%d'%nsam)
	sam = np.asarray([mps.generate_sample() for _ in range(nsam)])
	return sam

def statistic(current):
	""" Categorize and count the samples
	Return an numpy.record whose dtype=[('x',int),('f',int)]"""
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


if __name__ == '__main__':
	dataset = np.load('BStest/BSdata.npy').reshape(-1, 16)
	"""The binary number form of BS is stored in BSind.npy, with the identical order with BSdata.npy"""
	m = MPS_c(16)
	m.left_cano()
	m.designate_data(dataset)
	m.init_cumulants()

	m.cutoff = 5e-5
	m.descent_step_length = 0.05
	m.descent_steps = 10
	m.train(2)

	m.saveMPS('BS-',True)
	sam_zip = remember_zipper(m, 1000)
	# os.chdir('BS-MPS')
	np.save('sam_zip.npy', sam_zip)
	np.save('memo_zip.npy', statistic(sam_zip))
	sam_met = remember(m, 5000, 1000)
	np.save('sam_met.npy', sam_met)
	np.save('memo_met.npy', statistic(sam_met))
