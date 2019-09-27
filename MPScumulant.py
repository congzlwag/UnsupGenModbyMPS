# -*- coding: utf-8 -*-
"""
class MPS cumulant
@author: congzlwag
"""
import numpy as np
from numpy import ones, dot, zeros, log, asarray, save, load, einsum
from scipy.linalg import norm, svd
from numpy.random import rand, seed, randint
from time import strftime
import os
import sys

class MPS_c:
	def __init__(self, space_size):
		"""
		MPS class, with cumulant technique, efficient in DMRG-2
		Attributes:
			space_size: length of chain
			cutoff: truncation rate
			descenting_step_length: learning rate
			nbatch: number of batches

			bond_dimension: bond dimensions; 
				bond[i] connects i & i+1
			matrices: list of the tensors A^{(k)}
			merged_matrix:
				caches the merged order-4 tensor,
				when two adjacent tensors are merged but not decomposed yet
			current_bond: a multifunctional pointer
				-1: totally random matrices, need canonicalization
				in range(space_size-1): 
					if merge_matrix is None: current_bond is the one to be merged next time
					else: current_bond is the merged bond
				space_size-1: left-canonicalized
			cumulant: see init_cumulants.__doc__
			
			Loss: recorder of loss
			trainhistory: recorder of training history
		"""
		self.space_size = space_size
		self.cutoff = 0.01
		self.descenting_step_length = 0.1
		self.descent_steps = 10
		self.verbose = 1
		self.nbatch = 1

		# bond[i] connects i & i+1
		init_bondim = 2
		self.minibond = 2
		self.maxibond = 300
		self.bond_dimension = init_bondim * \
			ones((space_size,), dtype=np.int16) 
		self.bond_dimension[-1] = 1
		seed(1)
		self.matrices = [rand(self.bond_dimension[i - 1], 
				2, self.bond_dimension[i]) for i in range(space_size)]

		self.current_bond = -1 
		"""Multifunctional pointer
		-1: totally random matrices, need canonicalization
		in range(space_size-1): 
			if merge_matrix is None: current_bond is the one to be merged next time
			else: current_bond is the merged bond
		space_size-1: left-canonicalized
		"""
		self.merged_matrix = None

		self.Loss = []
		self.trainhistory = []

	def left_cano(self):
		"""
		Canonicalizing all except the rightmost tensor left-canonical
		Can be called at any time
		"""
		if self.merged_matrix is not None:
			self.rebuild_bond(True, kepbdm=True)
		if self.current_bond == -1:
			self.current_bond = 0
		for bond in range(self.current_bond,self.space_size - 1):
			self.merge_bond()
			self.rebuild_bond(going_right=True, kepbdm=True)

	def merge_bond(self):
		k = self.current_bond
		self.merged_matrix = np.einsum('ijk,klm->ijlm',self.matrices[k],
							self.matrices[(k+1) % self.space_size], order='C')

	def normalize(self):
		self.merged_matrix /= norm(self.merged_matrix)

	def rebuild_bond(self, going_right, spec=False, cutrec=False, kepbdm=False):
		"""Decomposition
		going_right: if we're sweeping right or not
		kepbdm: if the bond dimension is demanded to keep invariant compared with the value before being merged,
			when gauge transformation is carried out, this is often True.
		spec: if the truncated singular values are returned or not
		cutrec: if a recommended cutoff is returned or not
		"""
		assert self.merged_matrix is not None
		k = self.current_bond
		kp1 = (k+1)%self.space_size
		U, s, V = svd(self.merged_matrix.reshape((self.bond_dimension[
					  (k - 1) % self.space_size] * 2, 2 * self.bond_dimension[kp1])))
		
		if s[0]<=0.:
			print('Error: At bond %d Merged_mat happens to be all-zero.\nPlease tune learning rate.'%self.current_bond)
			raise FloatingPointError('Merged_mat trained to all-zero')

		if self.verbose:
			print("bond:", k)
		if self.verbose > 1:
			print(s)

		if not kepbdm:
			bdmax = min(self.maxibond, s.size)
			l = self.minibond
			while l < bdmax and s[l] >= s[0] * self.cutoff:
				l += 1
			#Found l: s[:l] dominate after cut off
		else:
			l = self.bond_dimension[k]
			#keep bond dimension

		if cutrec:
			if l >= bdmax:
				cut_recommend = -1.
			else:
				cut_recommend = s[l]/s[0]
		s = np.diag(s[:l])
		U = U[:, :l]
		V = V[:l, :]
		bdm_last = self.bond_dimension[k]
		self.bond_dimension[k] = l

		if going_right:
			V = dot(s, V)
			V /= norm(V)
		else:
			U = dot(U, s)
			U /= norm(U)

		if not kepbdm:
			if self.verbose > 1:
				print(self.bond_dimension)
			elif self.verbose >0:
				print('Bondim %d->%d'%(bdm_last,l))

		self.matrices[k] = U.reshape((self.bond_dimension[(k - 1) % self.space_size], 
										2, l))
		self.matrices[kp1] = V.reshape((l, 2, self.bond_dimension[kp1]))

		self.current_bond += 1 if going_right else -1
		self.merged_matrix = None

		if spec:
			if cutrec:
				return np.diag(s), cut_recommend
			else:
				return np.diag(s)
		else:
			if cutrec:
				return cut_recommend

	def designate_data(self, dataset):
		"""Before the training starts, the training set is designated"""
		self.data = dataset.astype(np.int8)
		self.batchsize = self.data.shape[0]//self.nbatch

	def init_cumulants(self):
		"""
		Initialize a cache for left environments and right environments, `cumulants'
		During the training phase, it will be kept unchanged that:
		1) len(cumulant)== space_size
		2) cumulant[0]  == ones((n_sample, 1))
		3) cumulant[-1] == ones((1, n_sample))
		4)  k = current_bond
			cumulant[j] = 	if 0<j<=k: A(0)...A(j-1)
							elif k<j<space_size-1: A(j+1)...A(space_size-1)
		"""
		if self.current_bond == self.space_size-1:
			#In this case, the MPS is left-canonicalized except the right most one, so the bond to be merged is space_size-2
			self.current_bond -= 1
		self.cumulants = [ones((self.data.shape[0], 1))]
		for n in range(0, self.current_bond):
			self.cumulants.append(einsum('ij,jik->ik',
					self.cumulants[-1], self.matrices[n][:,self.data[:,n],:]))
		right_part = [ones((1, self.data.shape[0]))]
		for n in range(self.space_size-1, self.current_bond+1, -1):
			right_part = [einsum('jil,li->ji',self.matrices[n][:,self.data[:,n]],right_part[0])] + right_part
		self.cumulants = self.cumulants + right_part

	def Give_psi_cumulant(self):
		"""Calculate the psi on the training set"""
		k = self.current_bond
		if self.merged_matrix is None:
			return einsum('ij,jik,kil,li->i',
					self.cumulants[k],self.matrices[k][:,self.data[:,k],:],
					self.matrices[k+1][:,self.data[:,k+1],:],self.cumulants[k+1])
		else:
			return einsum('ij,jik,ki->i',
					self.cumulants[k],self.merged_matrix[:,self.data[:,k],self.data[:,k+1],:],self.cumulants[k+1])

	def Show_Loss(self, append=True):
		"""Show the NLL averaged on the training set"""
		L = -log(np.abs(self.Give_psi_cumulant())**2).mean() #- self.data_shannon
		if append:
			self.Loss.append(L)
		if self.verbose > 0:
			print("Current loss:", L)
		return L

	def Calc_Loss(self, dat):
		"""Show the NLL averaged on an arbitrary set"""
		L = -log(np.abs(self.Give_psi(dat))**2).mean()
		if self.verbose > 0:
			print("Calculated loss:", L)
		return L

	def Gradient_descent_cumulants(self, batch_id):
		""" Gradient descent using cumulants, which efficiently avoids lots of tensor contraction!\\
			Together with update_cumulants, its computational complexity for updating each tensor is D^2
			Added by Pan Zhang on 2017.08.01
			Revised to single cumulant by Jun Wang on 20170802
		"""
		indx = range(batch_id*self.batchsize,(batch_id+1)*self.batchsize)
		states = self.data[indx]
		k = self.current_bond
		kp1 = (k+1)%self.space_size
		km1 = (k-1)%self.space_size
		left_vecs=self.cumulants[k][indx,:] # batchsize * D
		right_vecs=self.cumulants[kp1][:,indx] # D * batchsize

		phi_mat = einsum('ij,ki->ijk',left_vecs,right_vecs) # batchsize*D*D
		psi = einsum('ij,jik,ki->i',left_vecs,self.merged_matrix[:,states[:,k],states[:,kp1],:],right_vecs )# batchsize*D

		gradient = zeros([self.bond_dimension[km1], 2, 2, self.bond_dimension[kp1]])
		psi_inv = 1/psi
		if (psi==0).sum():
			print('Error: At bond %d, batchsize=%d, while %d of them psi=0.'%(self.current_bond, self.batchsize,(psi==0).sum()))
			print(np.argwhere(psi==0).ravel())
			print('Maybe you should decrease n_batch')
			raise ZeroDivisionError('Some of the psis=0')
		for i,j in [(i,j) for i in range(2) for j in range(2)]:
			idx = (states[:,k]==i) * (states[:,kp1]==j)
			gradient[:,i,j,:]=np.einsum('ijk,i->jk',phi_mat[idx,:,:],psi_inv[idx])*2
		gradient = gradient / self.batchsize - 2 * self.merged_matrix
		self.merged_matrix += gradient * self.descenting_step_length
		self.normalize()

	def update_cumulants(self,gone_right_just_now):
		"""After rebuid_bond, update self.cumulants.
		Bond has been rebuilt and self.current_bond has been changed,
		so it matters whether we have bubbled toward right or not just now
		"""
		k = self.current_bond
		if gone_right_just_now:
			self.cumulants[k] = einsum('ij,jik->ik',self.cumulants[k-1],self.matrices[k-1][:,self.data[:,k-1],:])
		else:
			self.cumulants[k+1] = einsum('jik,ki->ji',self.matrices[k+2][:,self.data[:,k+2],:],self.cumulants[k+2])

	def __bondtrain__(self, going_right, cutrec=False, showloss=False):
		"""Training on current_bond
		going_right & cutrec: see rebuild_bond
		showloss: whether Show_Loss is called.
		"""
		self.merge_bond()
		batch_start = randint(self.nbatch)
		# batch_start = 0
		self.batchsize = self.data.shape[0] // self.nbatch
		for n in range(self.descent_steps):
			self.Gradient_descent_cumulants(batch_id=(batch_start + n) %self.nbatch)
		# self.Show_Loss()#Before cutoff
		cut_recommend = self.rebuild_bond(going_right, cutrec=cutrec)
		self.update_cumulants(gone_right_just_now=going_right)
		if showloss:
			self.Show_Loss()
		if cutrec:
			return cut_recommend

	def train(self, Loops, rec_cut=True):
		"""Training over several epoches. `Loops' is the number of epoches"""
		for loop in range(Loops-1 if rec_cut else Loops):
			for bond in range(self.space_size - 2, 0, -1):
				self.__bondtrain__(False, showloss=(bond==1))
			for bond in range(0, self.space_size - 2):
				self.__bondtrain__(True, showloss=(bond==self.space_size-3))
			print("Current Loss: %.9f\nBondim:"% self.Loss[-1])
			print(self.bond_dimension)
			
		if rec_cut:
			#Now loop = Loops - 1
			cut_rec = [self.__bondtrain__(False, True)]
			for bond in range(self.space_size - 3, 0, -1):
				self.__bondtrain__(False, showloss=(bond==1))
			for bond in range(0, self.space_size - 2):
				cut_rec.append(self.__bondtrain__(True, True, bond==self.space_size-3))
			print("Current Loss: %.9f\nBondim:"% self.Loss[-1])
			print(self.bond_dimension)
		#All loops finished
		#print('Append History')
		self.trainhistory.append(
			[self.cutoff, Loops, self.descent_steps, self.descenting_step_length, self.nbatch])
		if rec_cut:
			if self.verbose > 2:
				print(asarray(cut_rec))
			cut_rec.sort(reverse=True)
			k = max(5,int(self.space_size*0.2))
			while k >= 0 and cut_rec[k] < 0:
				k -= 1
			if k >= 0:
				print('Recommend cutoff for next loop:', cut_rec[k])
				return cut_rec[k]
			else:
				print('Recommend cutoff for next loop:', 'Keep current value')
				return self.cutoff

	def saveMPS(self, prefix='', override=False):
		"""Saving all the information of the MPS into a folder
		The name of the folder is defaultly set as:
			prefix + strftime('MPS_%H%M_%d_%b'), about the moment you save it
		but you can also override the timestamp, which means the name will be:
			prefix + 'MPS'
		"""
		assert self.merged_matrix is None
		if not override:
			timestamp = prefix + strftime('MPS_%H%M_%d_%b')
		else:
			timestamp = prefix + 'MPS'
		try:
			os.mkdir('./' + timestamp + '/')
		except FileExistsError:
			pass
		os.chdir('./' + timestamp + '/')
		fp = open('MPS.log', 'w')
		fp.write("Present State of MPS:\n")
		fp.write("space_size=%d,\ncutoff=%1.5e,\tlr=%1.5e,\tnstep=%d\tnbatch=%d\n" %
				 (self.space_size, self.cutoff, self.descenting_step_length, self.descent_steps, self.nbatch))
		fp.write("bond dimension:\n")
		a = int(np.sqrt(self.space_size))
		if self.space_size % a == 0:
			a *= int(np.log10(self.bond_dimension.max()))+2
			fp.write(np.array2string(self.bond_dimension,precision=0,max_line_width=a))
		else:
			fp.write(np.array2string(self.bond_dimension,precision=0))
		try:
			fp.write("\nloss=%1.6e\n" % self.Loss[-1])
		except:
			pass
		save('Cutoff.npy', self.cutoff)
		save('Loss.npy', asarray(self.Loss))
		save('Bondim.npy', self.bond_dimension)
		save('TrainHistory.npy', asarray(self.trainhistory))
		save('CurrentBond.npy', self.current_bond)
		np.savez_compressed('Mats.npz', Mats=self.matrices)
		fp.write("cutoff\tn_loop\tn_descent\tlearning_rate\tn_batch\n")
		for history in self.trainhistory:
			fp.write("%1.2e\t%d\t%d\t\t%1.2e\t%d\n" % tuple(history))
		fp.close()
		os.chdir('../')

	def loadMPS(self, srch_pwd=None):
		"""Loading a MPS from directory `srch_pwd'. If it is None, then search at current working directory"""
		if srch_pwd is not None:
			oripwd = os.getcwd()
			os.chdir(srch_pwd)
		self.bond_dimension = load('Bondim.npy')
		self.space_size = len(self.bond_dimension)
		self.trainhistory = load('TrainHistory.npy').tolist()
		self.Loss = load('Loss.npy').tolist()
		try:
			self.current_bond = int(load('CurrentBond.npy'))
		except FileNotFoundError:
			self.current_bond = self.space_size - 2
			#most MPS are saved after a loop, when current_bond is space_size-2
		try:
			last = self.trainhistory[-1]
			try:
				self.cutoff, _, self.descent_steps, self.descenting_step_length, self.nbatch = last
			except ValueError:
				self.cutoff, _, self.descent_steps, self.descenting_step_length = last
			self.descent_steps = int(self.descent_steps)
			self.nbatch = int(self.nbatch)
		except:
			pass
		try:
			self.cutoff = float(load('Cutoff.npy'))
		except:
			pass
		try:
			self.matrices = list(np.load('Mats.npz')['Mats'])
		except:
			self.matrices = [load('Mat_%d.npy' % i) for i in range(self.space_size)]

		self.merged_matrix = None
		if srch_pwd is not None:
			os.chdir(oripwd)

	def Give_psi(self, states):
		"""Calculate the corresponding psi for configuration `states'"""
		if states.ndim == 1:
			states = states.reshape((1,-1))
		if self.merged_matrix is not None:
		# There's a merged tensor
			nsam = states.shape[0]
			k = self.current_bond
			kp1 = (k+1)%self.space_size
			left_vecs = np.ones((nsam, 1))
			right_vecs= np.ones((1, nsam))
			for i in range(0,k):
				left_vecs = einsum('ij,jik->ik',left_vecs,self.matrices[i][:,states[:,i],:])
			for i in range(self.space_size-1,k+1,-1):
				right_vecs = einsum('jik,ki->ji',self.matrices[i][:,states[:,i],:],right_vecs)
			return einsum('ik,kil,li->i',
					left_vecs, self.merged_matrix[:,states[:,k],states[:,kp1],:], right_vecs)
		else:
		# TT -- default status
			# try:
			left_vecs = self.matrices[0][0,states[:,0],:]
			# except IndexError:
			# 	print(self.matrices[0].shape, states)
			# 	sys.exit(-10)
			for n in range(1, self.space_size-1):
				left_vecs = einsum('ij,jil->il', left_vecs, self.matrices[n][:,states[:,n],:])
			return einsum('ij,ji->i',left_vecs,self.matrices[-1][:,states[:,-1],0])

	def Give_probab(self, states):
		"""Calculate the corresponding probability for configuration `states'"""
		return np.abs(self.Give_psi(states))**2

	def generate_sample(self, given_seg = None, *arg):
		"""
		Warning: This method has already been functionally covered by generate_sample_1, so it might be discarded in the future.
		Usage:
			1) Direct sampling: m.generate_sample()
			2) Conditioned sampling: m.generate_sample((l, r), array([s_l,s_{l+1},...,s_{r-1}]))
				array([s_l,s_{l+1},...,s_{r-1}]) is given, and (l,r) designates the location of this segment
		"""
		state = np.empty((self.space_size,), dtype=np.int8)
		if given_seg is None:
			if self.current_bond != self.space_size - 1:
				print("Warning: MPS should have been left canonicalized, when generating samples")
				self.left_cano()
				print("Left-canonicalized, but please add left_cano before generation.")
			vec = asarray([1])
			for p in range(self.space_size - 1, -1, -1):
				vec_act = dot(self.matrices[p][:, 1], vec)
				if rand() < (norm(vec_act) / norm(vec))**2:
					state[p] = 1
					vec = vec_act
				else:
					state[p] = 0
					vec = dot(self.matrices[p][:, 0], vec)
		else:
			l, r = given_seg
			#assign the given segment
			state[l:r] = arg[0][:]
			#canonicalization
			if self.current_bond > r-1:
				for bond in range(self.current_bond, r-2, -1):
					self.merge_bond()
					self.rebuild_bond(going_right=False, kepbdm=True)
			elif self.current_bond < l:
				for bond in range(self.current_bond, l):
					self.merge_bond()
					self.rebuild_bond(going_right=True, kepbdm=True)
			vec = self.matrices[l][:,state[l],:]
			for p in range(l+1, r):
				vec = dot(vec, self.matrices[p][:,state[p],:])
				vec /= norm(vec)
			for p in range(r, self.space_size):
				vec_act = dot(vec, self.matrices[p][:,1])
				# if rand() < (norm(vec_act) / norm(vec))**2:
				if rand() < norm(vec_act)**2:
					#activate
					state[p] = 1
					vec = vec_act
				else:
					#keep 0
					state[p] = 0
					vec = dot(vec, self.matrices[p][:,0])
				vec /= norm(vec)
			for p in range(l-1, -1, -1):
				vec_act = dot(self.matrices[p][:,1], vec)
				# if rand() < (norm(vec_act) / norm(vec))**2:
				if rand() < norm(vec_act)**2:
					state[p] = 1
					vec = vec_act
				else:
					state[p] = 0
					vec = dot(self.matrices[p][:,0],vec)
				vec /= norm(vec)
		return state

	def generate_sample_1(self, stat=None, givn_msk=None):
		"""
		This direct sampler generate one sample each time.
		We highly recommend to canonicalize the MPS such that the only uncanonical bit is given,
		because when conducting mass sampling, canonicalization will be an unnecessary overhead!
		Usage:
			If the generation starts from scratch, just keep stat=None and givn_msk=None;
			else please assign
				givn_msk: an numpy.array whose shape is (space_size,) and dtype=bool
				to specify which of the bits are given, and
				stat: an numpy.array whose shape is (space_size,) and dtype=numpy.int8
				to specify the configuration of the given bits, the other bits will be ignored.

		"""
		# <<<case: Start from scratch
		if stat is None or givn_msk is None or givn_msk.any()==False:
			if self.current_bond != self.space_size - 1:
				self.left_cano()
			state = np.empty((self.space_size,), dtype=np.int8)
			vec = asarray([1])
			for p in np.arange(self.space_size)[::-1]:
				vec_act = dot(self.matrices[p][:, 1], vec)
				if rand() < (norm(vec_act) / norm(vec))**2:
					state[p] = 1
					vec = vec_act
				else:
					state[p] = 0
					vec = dot(self.matrices[p][:, 0], vec)
			return state
		#case: Start from scratch>>>

		state = stat.copy()
		state[givn_msk==False] = -1
		givn_mask = givn_msk.copy()
		p = self.space_size-1
		while givn_mask[p] == False:
			p -= 1
		p_uncan = p
		# p_uncan points on the rightmost given bit

		"""Canonnicalizing the MPS into mix-canonical form that the only uncanonical tensor is at p_uncan
		There's a bit trouble, for the uncanonical tensor is not recorded.
		It can be on either the left or the right of current_bond, 
		so we firstly check whether the bits on both sides of current_bond are given or not
		"""
		bd = self.current_bond
		if givn_mask[bd]==False or givn_mask[bd+1]==False:
		# not both of the bits connected by current_bond are given, so we need to canonicalize the MPS
			if bd >= p_uncan:
				while self.current_bond >= p_uncan:
					self.merge_bond()
					self.rebuild_bond(False,kepbdm=True)
			else:
				while self.current_bond < p_uncan:
					self.merge_bond()
					self.rebuild_bond(False,kepbdm=True)
		"""Canonicalization finished
		From now on we should never operate the matrices in the sampling
		"""
		plft = 0
		while givn_mask[plft] == False:
			plft += 1
		# plft points on the leftmost given bit

		p = plft
		while p<self.space_size and givn_mask[p]:
			p += 1
		plft2 = p-1
		# Since plft is a given bit, there's a segment of bits that plft is in. plft2 points on the right edge of this segment

		# <<< If there's no intermediate bit that need to be sampled
		if plft2 == p_uncan:
			vec = self.matrices[plft][:,state[plft],:]
			for p in range(plft+1, plft2+1):
				vec = dot(vec, self.matrices[p][:,state[p],:])
				vec /= norm(vec)
			for p in range(plft2+1, self.space_size):
				vec_act = dot(vec, self.matrices[p][:,1])
				nom = norm(vec_act)
				if rand() < nom**2:
					#activate
					state[p] = 1
					vec = vec_act/nom
				else:
					#keep 0
					state[p] = 0
					vec = dot(vec, self.matrices[p][:,0])
					vec /= norm(vec)
			for p in np.arange(plft)[::-1]:
				vec_act = dot(self.matrices[p][:,1], vec)
				nom = norm(vec_act)
				if rand() < nom**2:
					state[p] = 1
					vec = vec_act/nom
				else:
					state[p] = 0
					vec = dot(self.matrices[p][:,0],vec)
					vec /= norm(vec)
			# assert (state!=-1).all()
			return state
		# >>>
		
		"""Dealing with the intermediated ungiven bits, sampling from plft2 to p_uncan. Only ungiven bits are sampled, of course.
		Firstly, we need to prepare
			left_vec: a growing ladder-shape TN, accumulatedly multiplied from plft to the right edge of the given segment plft is in.
			right_vecs: list of ladder-shape TNs
				right_vecs[p] is the TN accumulately multiplied from p_uncan to p (including p)
		"""
		left_vec = einsum("kj,kl->jl",self.matrices[plft][:,state[plft]],self.matrices[plft][:,state[plft]])
		left_vec /= np.trace(left_vec)
		for p in range(plft+1, plft2+1):
			left_vec = einsum("pl,lq->pq",einsum("jl,jp->pl",left_vec,self.matrices[p][:,state[p]]),self.matrices[p][:,state[p]])
			left_vec /= np.trace(left_vec)

		right_vecs = np.empty((self.space_size), dtype=object)
		p = p_uncan
		right_vecs[p] = einsum("ij,kj->ik",self.matrices[p][:,state[p]],self.matrices[p][:,state[p]])
		right_vecs[p] /= np.trace(right_vecs[p])
		p -= 1
		while p > plft2:
			if givn_mask[p]:
				right_vecs[p] = einsum("qk,pk->pq",self.matrices[p][:,state[p]],einsum("pi,ik->pk",self.matrices[p][:,state[p]],right_vecs[p+1]))
			else:
				right_vecs[p] = einsum("qjk,pjk->pq",self.matrices[p],einsum("pji,ik->pjk",self.matrices[p],right_vecs[p+1]))
			right_vecs[p] /= np.trace(right_vecs[p])
			p -= 1
		
		# Secondly, sample the intermediate bits
		p = plft2+1
		while p <= p_uncan:
			if not givn_mask[p]:
				prob_marg = einsum("pq,pq",
					einsum("pil,liq->pq",einsum("jl,jip->pil",left_vec,self.matrices[p]),self.matrices[p]),right_vecs[p+1])
				left_vec_act = einsum("pl,lq->pq",einsum("jl,jp->pl",left_vec,self.matrices[p][:,1]),self.matrices[p][:,1])
				prob_actv = einsum("pq,pq",left_vec_act,right_vecs[p+1])
				if rand()<prob_actv/prob_marg:
					state[p] = 1
					left_vec = left_vec_act
				else:
					state[p] = 0
					left_vec = einsum("pl,lq->pq",einsum("jl,jp->pl",left_vec,self.matrices[p][:,0]),self.matrices[p][:,0])
				givn_mask[p] = True
			else:
				left_vec = einsum("pl,lq->pq",einsum("jl,jp->pl",left_vec,self.matrices[p][:,state[p]]),self.matrices[p][:,state[p]])
			left_vec /= np.trace(left_vec)
			p += 1

		# Sampling the ungiven segment that connects to the right end
		while p < self.space_size:
			left_vec_act = einsum("pl,lq->pq",einsum("jl,jp->pl",left_vec,self.matrices[p][:,1]),self.matrices[p][:,1])
			prob_actv = np.trace(left_vec_act)
			if rand() < prob_actv:
				state[p] = 1
				left_vec = left_vec_act
				left_vec /= prob_actv
			else:
				state[p] = 0
				left_vec = einsum("pl,lq->pq",einsum("jl,jp->pl",left_vec,self.matrices[p][:,0]),self.matrices[p][:,0])
				left_vec /= np.trace(left_vec)
			givn_mask[p] = True
			p+=1
		# Sample the ungiven segment that connects to the left end
		if plft == 0:
		# bit 0 is given
			return state
		# else recursively generate
		return self.generate_sample_1(state,givn_mask)

	def proper_cano(self, target_bond, update_cumulant):
		"""Gauge Transform the MPS into the mix-canonical form that:
		the only uncanonical tensor is either target_bond or target_bond+1. Both are accepted.
		"""
		if self.current_bond == target_bond:
			return
		else:
			direction = 1 if self.current_bond<target_bond else -1
			for b in range(self.current_bond, target_bond, direction):
				self.merge_bond(b)
				self.rebuild_bond(going_right=(direction==1), kepbdm=True)
				if update_cumulant:
					self.update_cumulants(direction==1)
