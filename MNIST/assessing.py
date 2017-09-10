# -*- coding: utf-8 -*-
from sys import path, argv
path.append('../')
import numpy as np
from numpy.random import rand, shuffle, randint
from numpy.linalg import norm
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt

from MPScumulant import MPS_c

def arrayplot(imgs, name):
	"""plot the imgs (of shape (n_images, length_a, length_a)) in an array and save them as name"""
	nn = imgs.shape[0]
	ncol = int(np.sqrt(nn))
	while nn%ncol:
		ncol -= 1
	fig, axs = plt.subplots(nn//ncol, ncol, figsize=(ncol,nn//ncol))
	for ii in range(nn):
		ax = axs.ravel()[ii]
		ax.matshow(imgs[ii], cmap = "gray_r")
		ax.set_axis_off()
	plt.subplots_adjust(left=0, bottom=0, right=1.0, top=1.0, hspace=0.2, wspace=0.3)
	plt.savefig(name)

def annoise(imgs, level):
	"""With the probability indicated by level, randomly flip each pixel in each image"""
	flippers = rand(imgs.size) #uniform on (0,1)
	flippers = flippers < level
	flippers.shape = imgs.shape
	return imgs^flippers

def make_mask(**kwarg):
	"""Make a given_mask in order for the application of MPS_c.generate_sample_1"""
	given_mask = np.zeros((length_a,length_a),dtype=bool)
	if 'left' in kwarg.keys():
		nn = kwarg['left']
		given_mask[:,nn:] = True
		descript = "left%d"%nn
	elif 'right' in kwarg.keys():
		nn = kwarg['right']
		given_mask[:,:length_a-nn] = True
		descript = "riht%d"%nn
	elif 'head' in kwarg.keys():
		nn = kwarg['head']
		given_mask[nn:,:] = True
		descript = "head%d"%nn
	elif 'tail' in kwarg.keys():
		nn = kwarg['tail']
		given_mask[:length_a-nn,:] = True
		descript = "tail%d"%nn
	return given_mask, descript


def denoise_clamp_bars(imgs):
	"""Denoise the imgs in three steps
		Reconstruct the top eleven rows from the rest;
		Reconstruct row 11-16 from the rest;
		Reconstruct the bottom eleven rows from the rest"""
	states = imgs.reshape((-1,m.space_size))

	gmasks = np.zeros((3,length_a,length_a),dtype=bool)
	gmasks[0,11:] = True
	gmasks[1,:11] = True
	gmasks[1,17:] = True
	gmasks[2,:17] = True

	denose = []
	denose.append([m.generate_sample_1(s, gmasks[0].ravel()) for s in states])
	denose.append([m.generate_sample_1(s, gmasks[1].ravel()) for s in states])
	denose.append([m2.generate_sample_1(s,gmasks[2].ravel()) for s in states])
	denose = np.asarray(denose).reshape(3,-1,length_a,length_a)
	for i in range(3):
		arrayplot(denose[i],'denoise_bar_%s-%d.eps'%(nspercent, i))

	answer = np.concatenate((denose[0,:,:11],denose[1,:,11:17],denose[2,:,17:]),1)
	np.savez('denois_bar_%s.npz'%nspercent,l0_10=denose[0],l11_16=denose[1],l17_27=denose[2], combo=answer)
	return answer


def complement(imgs, givn_lin_rang):
	states = imgs.reshape(imgs.shape[0],-1)
	bit_rang = givn_lin_rang[0]*length_a, givn_lin_rang[1]*length_a
	ans = np.asarray([m.generate_sample(bit_rang, state[bit_rang[0]:bit_rang[1]]) for state in states])
	np.save('reconstr.npy', ans)
	return ans.reshape(-1,length_a, length_a)


def test_denoise(imgs,level):
	global nsrate
	global nspercent
	nsrate = level
	nspercent = str(int(100*nsrate)).zfill(2)
	print('Noise: %s percent'%(nspercent))
	annoised = np.load('../annoised%s.npy'%nspercent)
	answer = denoise_clamp_bars(annoised)
	arrayplot(answer, 'denoised.pdf')

def test_complement_1(imgs, given_mask, descript):
	"""Notice: CANONICALIZE the MPS firstly will be significantly helpful in efficiency, especially when reconstructin right or tail"""
	m.verbose = 0
	states = imgs.reshape(-1,m.space_size)
	ans = np.asarray([m.generate_sample_1(st, given_mask.ravel()) for st in states])
	ans.shape =(-1,length_a, length_a)
	np.save('reconstr_%s.npy'%(descript), ans)
	ans[:,given_mask] *= 2
	arrayplot(ans, 'reconstr_%s.pdf'%(descript))
	

if __name__ == '__main__':
	images = np.load('../otherimg.npy')
	length_a = int(np.sqrt(images.shape[1]))
	images.shape=(-1,length_a,length_a)

	lp = int(argv[1])
	m = MPS_c(length_a**2)
	m.loadMPS('../Loop%dMPS'%lp)
	print('Using: Loop%dMPS'%lp)
	m.verbose = 0

	"""m2 is the result of gauge transformation of m.
	The only uncanonical tensor in m2 is at the middle."""
	# m2 = MPS_c(length_a**2)
	# m2.loadMPS('../Loop%d_middleMPS'%lp)
	# print('Using: Loop%d_middleMPS'%lp)
	# m2.verbose = 0

	nimg = 20

	gmask, descr = make_mask(left=10)
	test_complement_1(images, gmask, descr)