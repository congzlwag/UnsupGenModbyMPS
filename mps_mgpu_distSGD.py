#!/usr/bin/env python3.6
"""
Multi gpu version, 2018.02.26 23:00 copied from mps_gpu_lowmem.py
02.28:  Single-GPU chunk-version is finished, and the file is renamed to mps_chunks.py
03.05:  Multiple-GPU version is finished.
03.06:  nll_history[] is added.
03.06:  some variables have been renamed, MPS.init() has been re-organized in a more elegant way,           which is also easy for saving/loading
03.06:  saving/loading using pickle. 
        saving is carried out in training() 
        loading fills the mps object, notice that after loading, self.broadcasting needs to be called to initialize data on each gpu.
03.10:  ztmp, _gpu_queue are redundant and removed
        altered idx_2site to ByteTensor, used as mask
        added self._tensortype to unify FloatTensor and DoubleTensor
03.10:  Removed unnecessary clone of self.newcumu
        Assigned self.newcumu and self.gradients to None after each usage
03.10:  Added sampling method
03.18:  Added attribute grad_mode
04.18:  Added attribute n_step
"""
#
#{{{ import
from __future__ import print_function
import numpy as np
from numpy.random import rand
import torch
import time
import sys,os
from multiprocessing.dummy import Pool as ThreadPool
import queue
import pickle
#}}}
#{{{ class MPS:
class MPS:
    """
    Matrix product states for generative modeling.
    """
    def __init__(self,data,batch_size,gpu_ids=[],use_double=False,init_bondim=2,data_test=[],fname="SGDresult",chunk_size=2000):
        """ 
        """
# Principles for storage: 1. variables for reading on GPU and variables for writing on CPU 2. Main thread always controls CPU and only one GPU for doing SVD and merging, all GPUs are used only in spawned threads.
# On CPU: self.tensors, self.cumulants, self.psis, self.psis_test, self.bondims, self.merged_tensor
# On GPU: self.data, self.data_test, self.merged_tensor_BC
# number of GPU cannot be changed
# batchsize & chunksize can be changed
        self._gpu_ids = gpu_ids # a list of gpu ids to use
        self._ngpu = len(gpu_ids)
        self.main_gpuid = self._gpu_ids[0] # id of gpu used for the main thread
        if use_double:
            self._tensortype = torch.DoubleTensor
            self.epsi = 1e-300
        else:
            self._tensortype = torch.FloatTensor
            self.epsi = 1e-26
        
        self.n=data.shape[1] # number of spins
        self.m=data.shape[0] # number of training samples
        self.m_test = len(data_test)
        assert self.m%len(gpu_ids)==0
        assert self.m_test%len(gpu_ids)==0
        self.m_gpu = self.m // len(gpu_ids)
        if self.m_test > 0:
            self.m_test_gpu = self.m_test // len(gpu_ids)

        self.data_cpu = data.type(torch.LongTensor).cpu()
        if(self.m_test > 0):
            self.data_test_cpu = data_test.type(torch.LongTensor).cpu()

        self.training_method=1 # 1: cumulants, default 2: slower one
        self.cutoff = 0.
        self.learning_rate = 0.01
        self.n_step = 1
        self.verbose = True
        self.converge_crit=0.01
        self.loop=0

        self.batch_size = self.m
        self.set_batchsize(batch_size)
        self.set_chunksize(chunk_size)
        self._build_idx()
        self.distribute()

        self.min_bondim = 2
        self.max_bondim = 200
        bondims = np.asarray([2**min(i,self.n-1-i,np.log2(init_bondim)) for i in range(self.n)],np.int16)
        self.bondims = torch.from_numpy(bondims)
        self.bondims.clamp_(self.min_bondim, self.max_bondim)
        self.bondims[-1] = 1

        self.randgn = np.random.RandomState(1996)
        self.seqSGD = False
        # Random initialization
        self.tensors= [torch.randn([self.bondims[i-1],2,self.bondims[i]]).type(self._tensortype) for i in range(self.n)]
        # # Prescribed 0^N
        # self.tensors = []
        # for i in range(self.n):
        #     tens = torch.zeros([self.bondims[i-1],2,self.bondims[i]]).type(self._tensortype)
        #     tens[0,0,0] = 1
        #     self.tensors.append(tens)
        # # Prescribed (0^N+1^N)
        # self.tensors = []
        # for i in range(self.n):
        #     tens = torch.zeros([self.bondims[i-1],2,self.bondims[i]]).type(self._tensortype)
        #     tens[0,0,0] = 1
        #     if i == 0 :
        #         tens[0,1,1] = 1
        #     elif i == self.n-1:
        #         tens[1,1,0] = 1
        #     else:
        #         tens[1,1,1] = 1
        #     self.tensors.append(tens)
        # # Prescribed (0+1)^N
        # self.tensors = []
        # for i in range(self.n):
        #     tens = torch.zeros([self.bondims[i-1],2,self.bondims[i]]).type(self._tensortype)
        #     tens[0,:,0] = 1
        #     self.tensors.append(tens)
        self.merged_tensor=self._tensortype() #self.merged_tensor will be on cpu and gid (in merge_bond() and rebuild_bond() for SVD), self.merged_tensor_BC will be the broadcasted version on each GPU
        self.tensor_bc=[]
        self.merged_tensor_BC=[] # broadcast version of merged_tensor, will be put on each GPU.

        self.grad_mode='plain' 
        """
        plain: original GD; 
        norm0.25: normalize with g.norm()/(g.numel()**0.25); 
        norm0.5: normalize with g.norm()/(g.numel()**0.5);
        elementwise: elementwise divide by (g**2+5e-6)**0.5
        """
        self.gradients=None # storing gradients in each gpu, will be used in gradient_descent()
        self.current_bond=-1 # indicating that it's random
        self.batch_head=0
        self.going_right=True
        print("n=",self.n,"m=",self.m,"batch_size=",self.batch_size)
        
        sys.stdout.write("left_canonical ");sys.stdout.flush();t1=time.time()
        self.left_canonical()
        sys.stdout.write("%.2f s.\n"%(time.time()-t1))
        sys.stdout.write("Initiate NLL ");sys.stdout.flush();t1=time.time()
        self.recompute_psi()
        self.nll = self.compute_nll()
        self.nll_test = 0
        if self.m_test>0:
            self.recompute_psi_test()
            self.nll_test = self.compute_nll_test()
        print("#%d nll=%.3f\tnll_test=%.3f\tmax_bondim=%d\t%.2f Sec."%(self.loop,self.nll,self.nll_test,max(self.bondims),time.time()-t1))
        self.fname = fname+'.dat'
        self.nll_history=np.array([(self.loop,self.nll,self.nll_test)],dtype=[('loop','i'),('train','d'),('test','d')])
        with open(self.fname,'a') as fp:
            print('# nll_trn nll_tst max(bdm)\tduration\n%d %.3f %.3f %d'%(self.loop,self.nll,self.nll_test,max(self.bondims)),file=fp)
        with open('hyper.txt','a') as fp:
            print('max_bdm\tcut\tlr\tnstp\tbhead\tmbatch\tngpu\tgmode',file=fp)
#}}}
#{{{def set_batchsize(self,batch_size):
    def set_batchsize(self,batch_size):
        bsiz_old = self.batch_size
        self.batch_size_gpu = int(min(batch_size,self.m)//self._ngpu)
        self.batch_size = self.batch_size_gpu*self._ngpu
        self.learning_rate *= float(self.batch_size)/bsiz_old
        if hasattr(self, "chunk_size"):
            self.chunk_size = min(self.chunk_size, self.batch_size_gpu)
        # if hasattr(self, 'batch_idx') and self.batch_idx.numel() != self.batch_size_gpu + self.m_gpu:
        #     self.batch_idx = torch.from_numpy(np.concatenate((np.arange(self.m_gpu),np.arange(self.batch_size_gpu)))).type(torch.LongTensor)
#}}}
#{{{def shrink_chunk(self, chunk_size):
    def set_chunksize(self, chunk_size):
        self.chunk_size = min(chunk_size, self.batch_size_gpu)
        # self.chunk_nd = np.append(np.arange(0,self.batch_size_gpu,self.chunk_size),self.batch_size_gpu) #nodes for chunk
        # if self.m_test > 0:
        #     self.chunk_nd_test =np.append(np.arange(0,self.m_test_gpu,self.chunk_size),self.m_test_gpu)
#}}}
    def alterGPU(self, gpu_ids):
        self._gpu_ids = gpu_ids
        self._ngpu = len(gpu_ids)
        self.main_gpuid = gpu_ids[0]
        self.m_gpu = int(self.m // self._ngpu)
        if self.m_test > 0:
            self.m_test_gpu = int(self.m_test // self._ngpu)
        self.set_batchsize(self.batch_size)
#{{{def _build_idx(self):
    def _build_idx(self):
        print("Building the 2-site index:",end='',flush=True);tt=time.time()
        # self.idx_cpu=torch.arange(0,self.m).view(self._ngpu,-1).type(torch.LongTensor).pin_memory()
        # if self.m_test>0:
            # self.idx_test_cpu=torch.arange(0,self.m_test).view(self._ngpu,-1).type(torch.LongTensor).pin_memory()
        # self.idx_2sites_cpu[bond, i, j, :] is a vector of shape (self.m,) which indicates whether the sample meets sample[bond]==i and sample[bond+1]==j
        self.idx_2sites_cpu=torch.ByteTensor(self.n-1,2,2,self.m).pin_memory()
        for bond in range(self.n-1):
            for i in range(2):
                for j in range(2):
                    self.idx_2sites_cpu[bond,i,j,:] = ((self.data_cpu[:,bond]==i)*(self.data_cpu[:,bond+1]==j)).cpu()
        print("%.2f Sec."%(time.time()-tt))
#}}}
#{{{def _broadcast_idx(self):
    # def _broadcast_chunk_pt(self):
    #     assert self.batch_head + self.batch_size_gpu <= self.m_gpu
        # self._chunk_pt = torch.cuda.comm.broadcast(torch.LongTensor([0,self.chunk_size]), self._gpu_ids)
#}}}
#{{{def distribute(self):
    def distribute(self):
        self.data =torch.cuda.comm.scatter(self.data_cpu,self._gpu_ids)# declare data as long tensor for indexing purpuse, then broadcast it to all gpus
        # self.idx_gpu=torch.comm.broadcast(torch.arange(self.m_gpu+self.batch_size_gpu).type(torch.LongTensor), self._gpu_ids)
        if not hasattr(self,"idx_2sites_cpu"):
            self._build_idx()
        self.idx_2sites=torch.cuda.comm.scatter(self.idx_2sites_cpu,self._gpu_ids,dim=3)
        if self.m_test>0:
            self.data_test=torch.cuda.comm.scatter(self.data_test_cpu,self._gpu_ids)# declare data as long tensor for indexing purpuse, then broadcast it to all gpus
            # self.idx_test_gpu=torch.comm.broadcast(torch.arange(self.m_test_gpu).type(torch.LongTensor), self._gpu_ids)
        for gidx in range(self._ngpu):
            print("On GPU #%d, training data shape"%self._gpu_ids[gidx], tuple(self.data[gidx].shape), end=', ')
            if self.m_test>0:
                print("test data shape",tuple(self.data_test[gidx].shape))
            else:
                print("no test data")
        self.psis=torch.ones(self._ngpu,self.m_gpu).type(self._tensortype).pin_memory()
        #notice that I do not want to put psis into GPU, as it is convenient to share the memory across threads, and it is easy to compute NLL based on self.psis
        if(self.m_test >0):
            self.psis_test=torch.ones(self._ngpu,self.m_test_gpu).type(self._tensortype).pin_memory()
#}}}

#{{{def left_canonical(self):
    def left_canonical(self):
        if self.current_bond < 0:
            self.current_bond = 0
        # elif self.going_right:
        #     self.current_bond += 1
        self.going_right=True
        while self.current_bond < self.n-1:
            self.merge_bond()
            self.rebuild_bond(True)
            self.current_bond += 1
#}}}
#{{{def merge_bond(self):
    def merge_bond(self):
        #This function uses single gpu, self.main_gpuid, for doing contractions.
        assert(self.current_bond>=0 and self.current_bond<self.n)
        if self.current_bond == self.n-1:
            self.current_bond -= 1
        d=self.bondims[self.current_bond]
        self.merged_tensor=torch.mm(self.tensors[self.current_bond].cuda(self.main_gpuid).view(-1,d),\
                            self.tensors[self.current_bond+1].cuda(self.main_gpuid).view(d,-1))\
                            .view(self.bondims[self.current_bond-1],2,2,self.bondims[self.current_bond+1])
#}}}
#{{{def rebuild_bond(self,fix_bondim=False):
    def rebuild_bond(self,fix_bondim=False):
        """rebuild merged_tensor into two self.tensors{bl} and
        self.merged_tensor will be set to [] at the end of this funciton"""
        dl=self.bondims[self.current_bond-1]
        d=self.bondims[self.current_bond]
        dr=self.bondims[self.current_bond+1]
        self.merged_tensor=self.merged_tensor.view(dl*2,dr*2)
        try:
            [U,s,V]=torch.svd(self.merged_tensor)
        except:
            print("GPU SVD does not converge, sending it to CPU")
            [U,s,V]=torch.svd(self.merged_tensor.cpu())
            U=U.cuda(self.main_gpuid)
            s=s.cuda(self.main_gpuid)
            V=V.cuda(self.main_gpuid)
            print("CPU svd() finished")
        if not fix_bondim:
            d=torch.sum(s>self.cutoff*s[0])
            d=max(min(d,self.max_bondim),self.min_bondim)
        U=U[:,:d]
        S=torch.diag(s[:d])
        V=V[:,:d].t()
        if(self.going_right):
            V=torch.mm(S,V)
            V/=V.norm()
        else:
            U=torch.mm(U,S)
            U/=U.norm()
        self.tensors[self.current_bond]=U.contiguous().view(dl,2,d).cpu()
        self.bondims[self.current_bond]=d
        self.tensors[self.current_bond+1]=V.contiguous().view(d,2,dr).cpu()
        self.merged_tensor=None
        return s[:d]
#}}}

#{{{def train(self,n_loops):
    def train(self):
        assert self.grad_mode in ["plain","norm0.25","norm0.5","elementwise"]
        self.loop += 1
        t1=time.time()
        svd_time=0
        other_time=0
        if self.seqSGD:
            if self.batch_head > self.m_gpu-self.batch_size_gpu:
                self.batch_head = self.m_gpu-self.batch_size_gpu
        else:
            self.batch_head = self.randgn.randint(self.m_gpu - self.batch_size_gpu + 1)
        print("Batch head=%d, init cumulants..."%(self.batch_head), end='',flush=True)
        self.init_cumulants()
        print("%.2f Sec."%(time.time()-t1))
        self.going_right=False
        for bond in np.concatenate((np.arange(self.n-2,0,-1),np.arange(0,self.n-2))).astype(np.uint16):
            print("\r bond #%d / %d "%(bond,self.n), end='', flush=True)
            t0=time.time()
            self.current_bond=bond
            if(bond==0):
                self.going_right=True
            self.merge_bond()
            for s in range(self.n_step):
                self.gradient_descent()
                if s <self.n_step-1:
                    self.merged_tensor /= self.merged_tensor.norm()
            # Since there's only one step, no need to normalize merged_tensor
            other_time += time.time()-t0; t0=time.time()
            self.rebuild_bond(False)
            svd_time += time.time()-t0; t0=time.time()
            self.update_cumulants()
            other_time += time.time()-t0
        t0=time.time()
        sys.stdout.write("\r                                                       \r")
        self.recompute_psi()
        nll_new=self.compute_nll()
        if(self.m_test >0):
            self.recompute_psi_test()
            self.nll_test=self.compute_nll_test()
        other_time += time.time()-t0
        sys.stdout.write("\r                                                       \r")
        print("#%d batch_head=%d\tnll=%.3f\tnll_test=%.3f\tmax_bondim=%d\t%.2f Sec., SVD %.2f, other %.2f"%(self.loop, self.batch_head, nll_new,self.nll_test,max(self.bondims),time.time()-t1,svd_time,other_time))
        if abs(nll_new-self.nll) < self.converge_crit:
            print("Converged")
            not_converged=False
        else:
            not_converged=True
        self.nll=nll_new
        self._log(time.time()-t1)
        if self.seqSGD:
            self.batch_head += self.batch_size_gpu
            if self.batch_head >= self.m_gpu:
                self.batch_head = 0
        return not_converged
#}}}
#{{{def jobs_threading(self,func):
    def _jobs_threading(self,func):
        self._job_queue=queue.Queue()
        pool=ThreadPool(self._ngpu)
        jobs=[]
        for gidx in range(self._ngpu):
            jobs.append(gidx)
            self._job_queue.put(gidx)
        pool.map(func,jobs)
        pool.close()
        while(not self._job_queue.empty()):
            time.sleep(0.01)
        else:
            assert self._job_queue.empty()
#}}}
#{{{def _worker_init_cumulants(self,gidx):
    def _worker_init_cumulants(self,gidx):
        gpuid=self._gpu_ids[gidx]
        states = self.data[gidx][self.batch_head:self.batch_head+self.batch_size_gpu, self.current_bond]
        chunk_begin, chunk_end = (0, self.chunk_size)
        while chunk_begin < self.batch_size_gpu:
            state_loc=states[chunk_begin:chunk_end]
            self.cumulants[self.current_bond+1][gidx,chunk_begin:chunk_end,:,:]=\
                    (self.cumulants[self.current_bond][gidx,chunk_begin:chunk_end,:,:].cuda(gpuid) \
                    @ self.tensor_bc[gidx][:,state_loc,:].permute(1,0,2)).cpu()
            self.cumulants[self.current_bond+1][gidx,chunk_begin:chunk_end,:,:] /= \
                max(self.cumulants[self.current_bond+1][gidx,chunk_begin:chunk_end].abs().max(),1e-6)
            ### normalize the psi values, for saving the precison. 
            ### It works together with adding small values for resolving the non psi issue 
            chunk_begin = chunk_end
            chunk_end = min(self.batch_size_gpu, chunk_end+self.chunk_size)
        self._job_queue.get()
#}}}    
#{{{ def init_cumulants(self,batch):
    def init_cumulants(self):
        """
        self.cumulants are created here for each batch, and stored in cpu
        size of self.cumulants is [cumulants[0],cumulants[1],...,cumulants[n]], each of which is a vector (stored as torch.tensor) of length batch_size.
        """
        self.cumulants=[] # its length is self.n
        self.cumulants.append(torch.ones(self._ngpu,self.batch_size_gpu,1,1).type(self._tensortype).pin_memory())
        for site in range(self.n-2): # n-2 for two-site update, notice that this would be n-1 for single-site update
            self.current_bond=site
            self.cumulants.append(torch.ones(self._ngpu,self.batch_size_gpu,1,self.bondims[site]).type(self._tensortype).pin_memory())
            self.tensor_bc=torch.cuda.comm.broadcast(self.tensors[site],self._gpu_ids)
            self._jobs_threading(self._worker_init_cumulants)
        self.cumulants.append(torch.ones(self._ngpu,self.batch_size_gpu,1,1).type(self._tensortype).pin_memory())
        self.tensor_bc = None
#}}}
#{{{def _worker_update_cumulants(self,gidx):
    def _worker_update_cumulants(self,gidx):
        gpuid= self._gpu_ids[gidx]
        btmp = self.current_bond+1 if self.going_right else self.current_bond
        if self.going_right:
            states_loc = self.data[gidx][self.batch_head:self.batch_head+self.batch_size_gpu,self.current_bond]
        else:
            states_loc = self.data[gidx][self.batch_head:self.batch_head+self.batch_size_gpu,self.current_bond+1]
        chunk_begin, chunk_end = (0, self.chunk_size)
        while chunk_begin < self.batch_size_gpu:
            if(self.going_right):
                newcumu = self.cumulants[self.current_bond][gidx,chunk_begin:chunk_end].cuda(gpuid)\
                         @ (self.tensor_bc[gidx][:,states_loc[chunk_begin:chunk_end]].permute(1,0,2))
            else:
                newcumu = self.tensor_bc[gidx][:,states_loc[chunk_begin:chunk_end]].permute(1,0,2)\
                         @ self.cumulants[self.current_bond+1][gidx,chunk_begin:chunk_end].cuda(gpuid)
            newcumu /= max(newcumu.abs().max(),1e-6) ## normalize among this chunk
            self.cumulants[btmp][gidx,chunk_begin:chunk_end]=newcumu.cpu()
            chunk_begin = chunk_end
            chunk_end = min(self.batch_size_gpu, chunk_end+self.chunk_size)
        self._job_queue.get()
#}}}
#{{{def update_cumulants(self):
    def update_cumulants(self):
        """
        Update cumulants for after one tensor is learnt in the current batch.
        """
        if(self.going_right):
            self.tensor_bc=torch.cuda.comm.broadcast(self.tensors[self.current_bond],self._gpu_ids)
            self.cumulants[self.current_bond+1]=self._tensortype(self._ngpu,self.batch_size_gpu,1,self.bondims[self.current_bond]).pin_memory()
        else:
            self.tensor_bc=torch.cuda.comm.broadcast(self.tensors[self.current_bond+1],self._gpu_ids)
            self.cumulants[self.current_bond]=self._tensortype(self._ngpu,self.batch_size_gpu,self.bondims[self.current_bond],1).pin_memory()
        self._jobs_threading(self._worker_update_cumulants)
        self.tensor_bc = None
        # btmp = self.current_bond+1 if self.going_right else self.current_bond
        # self.cumulants[btmp]=self.newcumu_batch
        # self.newcumu_batch = None
        ### normalize the psi values, for saving the precison. 
        ### It works together with adding small values for resolving the non psi issue
#}}}
#{{{def _worker_gradient_descent(self,gidx):
    def _worker_gradient_descent(self,gidx):
        gpuid=self._gpu_ids[gidx]
        chunk_begin, chunk_end = (0,self.chunk_size)
        local_idx_2sites_batch = self.idx_2sites[gidx][self.current_bond,:,:,self.batch_head:self.batch_head+self.batch_size_gpu]
        while chunk_begin < self.batch_size_gpu:
            cumu1=self.cumulants[self.current_bond][gidx,chunk_begin:chunk_end,:,:].cuda(gpuid)
            cumu2=self.cumulants[self.current_bond+1][gidx,chunk_begin:chunk_end,:,:].cuda(gpuid)
            converter  = torch.arange(chunk_end-chunk_begin).type(torch.LongTensor).cuda(gpuid)
            for i,j in [(i,j) for i in range(2) for j in range(2)]:
                idx = converter[local_idx_2sites_batch[i,j,chunk_begin:chunk_end]]
                if(idx.numel()==0):
                    continue
                lvecs=cumu1[idx]
                rvecs=cumu2[idx]
                psis = lvecs @ self.merged_tensor_BC[gidx][:,i,j,:].repeat(idx.numel(),1,1) @ rvecs
                psis += self.epsi  ### this is to resolve the zero psis issue, for the float-storage
                lvecs /= psis
                torch.addbmm(self.gradients[gidx][:,i,j,:],lvecs.permute(0,2,1),rvecs.permute(0,2,1),out=self.gradients[gidx][:,i,j,:])
            chunk_begin = chunk_end
            chunk_end = min(self.batch_size_gpu, chunk_end+self.chunk_size)
        self._job_queue.get()
#}}}    
#{{{def gradient_descent(self,batch):
    def gradient_descent(self):
        self.merged_tensor_BC=torch.cuda.comm.broadcast(self.merged_tensor,self._gpu_ids)
        self.gradients = [torch.zeros(self.bondims[self.current_bond-1],2,2,self.bondims[self.current_bond+1]).type(self._tensortype).cuda(gid) for gid in self._gpu_ids]
        self._jobs_threading(self._worker_gradient_descent)
        gradients = torch.cuda.comm.reduce_add(self.gradients, self.main_gpuid).cuda(self.main_gpuid)/self.batch_size-self.merged_tensor
        if self.grad_mode=='plain':
            pass
        elif self.grad_mode[:4]=='norm':
            if self.grad_mode=='norm0.25':
                gnorm = gradients.norm()/(gradients.numel()**0.25) # saddle killer?
            elif self.grad_mode=='norm0.5':
                gnorm = gradients.norm()/(gradients.numel()**0.5) 
            if gnorm < 1:
                gradients /= gnorm
        elif self.grad_mode=='elementwise':
            gradients /= (gradients**2+5e-6)**0.5
        self.merged_tensor += gradients*2.0*self.learning_rate
        self.gradients = None
        self.merged_tensor_BC = None
#}}}

#{{{def compute_nll(self):
    def compute_nll(self):
        return -torch.log(self.psis.abs()).mean()*2.0
#}}}
#{{{    def compute_nll_test(self):
    def compute_nll_test(self):
        return -torch.log(self.psis_test.abs()).mean()*2.0
#}}}

#{{{def _worker_recompute_psi(self,gidx):
    def _worker_recompute_psi(self,gidx):
        gpuid=self._gpu_ids[gidx]
        chunk_begin,chunk_end = (0,self.chunk_size)
        while chunk_begin < self.m_gpu:
            state=self.data[gidx][chunk_begin:chunk_end,:]
            psis=torch.ones(chunk_end-chunk_begin,1,1).type(self._tensortype).cuda(gpuid)
            for i in range(self.n):
                psis @= self.tensors[i].cuda(gpuid)[:,state[:,i],:].permute(1,0,2)
            self.psis[gidx,chunk_begin:chunk_end]=psis.view(-1,)
            chunk_begin = chunk_end
            chunk_end = min(chunk_end+self.chunk_size, self.m_gpu)
        self._job_queue.get()
#}}}
#{{{def recompute_psi(self):
    def recompute_psi(self):
        self._jobs_threading(self._worker_recompute_psi)
#}}}
#{{{def _worker_recompute_psi_test(self,gidx):
    def _worker_recompute_psi_test(self,gidx):
        gpuid=self._gpu_ids[gidx]
        chunk_begin,chunk_end = (0,min(self.chunk_size,self.m_test_gpu))
        while chunk_begin < self.m_test_gpu:
            state=self.data_test[gidx][chunk_begin:chunk_end,:]
            psis=torch.ones(chunk_end-chunk_begin,1,1).type(self._tensortype).cuda(gpuid)
            for i in range(self.n):
                psis @= self.tensors[i].cuda(gpuid)[:,state[:,i],:].permute(1,0,2)
            self.psis_test[gidx,chunk_begin:chunk_end]=psis.view(-1,)
            chunk_begin = chunk_end
            chunk_end = min(chunk_end+self.chunk_size, self.m_test_gpu)
        self._job_queue.get()
#}}}
#{{{def recompute_psi_test(self,batch):
    def recompute_psi_test(self):
        assert self.m_test >0
        self._jobs_threading(self._worker_recompute_psi_test)
#}}}
#{{{def _log(self):
    def _log(self, duration=0):
        self.nll_history = np.append(self.nll_history,np.array((self.loop,self.nll,self.nll_test),dtype=self.nll_history.dtype))
        with open(self.fname,"a") as fout:
            print("%d %.3f %.3f %d\t\t%.1f"%(self.loop,self.nll,self.nll_test,max(self.bondims),duration),file=fout,flush=True)
        with open('hyper.txt','a') as fp:
            #max_bdm\tcut\tlr\tnstp\tbhead\tmbatch\tngpu\tgmode
            print("%d\t%.2g\t%.3g\t%d\t%d\t%d\t%d\t%s"%\
            (self.max_bondim,self.cutoff,self.learning_rate,self.n_step,self.batch_head,self.batch_size,self._ngpu,self.grad_mode),file=fp)
#}}}
#{{{def generate_sample(self, stat=None, givn_mask=None):
    def generate_sample(self, stat=None, givn_mask=None, gid=0):
        """
        If stat is None, generate from scratch, return a direct sample;
        Else stat will be cloned and masked by givn_mask, sites where givn_mask==True will be clamped before sampling, return a conditional sample
        """
        gpu = self._gpu_ids[gid]
        # print("On GPU #%d"%gpu)
        if stat is None or givn_mask is None or givn_mask.any()==False:
            assert self.current_bond == self.n - 1
            state = np.empty((self.n,), dtype=np.uint8)
            vec = torch.ones(1).type(self._tensortype).cuda(gpu)
            for p in np.arange(self.n)[::-1]:
                vec_act = self.tensors[p][:, 1].cuda(gpu) @ vec
                nom = vec_act.norm()
                if rand() < nom**2:
                    state[p] = 1
                    vec = vec_act/nom
                else:
                    state[p] = 0
                    vec = self.tensors[p][:, 0].cuda(gpu) @ vec
                    vec /= vec.norm()
            return state
        if isinstance(stat, np.ndarray):
            state = stat.astype(np.uint8)
        else:
            state = stat.cpu().numpy().astype(np.int8)
        state[givn_mask==False] = -1
        # givn_mask = givn_mask.copy()

        given_idx = givn_mask.nonzero()[0]
        p_given_rightmost = given_idx.max()
        p_given_leftmost  = given_idx.min()

        p_unnorm = self.current_bond
        if self.current_bond != self.n-1 and self.going_right:
                p_unnorm += 1
        if givn_mask[p_unnorm] == False:
            targ_p_unnorm = given_idx[np.argmin(abs(given_idx-p_unnorm))]
            if p_unnorm > targ_p_unnorm:
                self.going_right = False
                while p_unnorm > targ_p_unnorm:
                    self.current_bond = p_unnorm-1
                    self.merge_bond()
                    self.rebuild_bond(True)
                    p_unnorm -= 1
                    print('Canonicalization: unnormalized site = #%d'%p_unnorm, end='\r', flush=True)
            elif p_unnorm < targ_p_unnorm:
                self.going_right = True
                while p_unnorm < targ_p_unnorm:
                    self.current_bond = p_unnorm
                    self.merge_bond()
                    self.rebuild_bond(True)
                    p_unnorm += 1
                    print('Canonicalization: unnormalized site = #%d'%p_unnorm, end='\r', flush=True)
        else:
            # print(givn_mask[bd],givn_mask[bd+1])
            print('No extra canonicalization!',end='')
        print('')
        # From now on matrices are fixed

        # print("Locating p_intermid_ungiven_leftmost")
        p = p_given_leftmost
        while p<self.n and givn_mask[p]:
            p += 1
        p_intermid_ungiven_leftmost = p
        # print('Given leftmost',p_given_leftmost, ', p_intermid_ungiven_leftmost', p_intermid_ungiven_leftmost, ', rightmost', p_given_rightmost)
        
        mid_mat_fromleft = self.tensors[p_given_leftmost][:,state[p_given_leftmost]].cuda(gpu)
        mid_mat_fromleft /= mid_mat_fromleft.norm()
        for p in range(p_given_leftmost+1, p_intermid_ungiven_leftmost):
            mid_mat_fromleft @= self.tensors[p][:,state[p]].cuda(gpu)
            mid_mat_fromleft /= mid_mat_fromleft.norm()

        if p_intermid_ungiven_leftmost < p_given_rightmost+1:
            right_vecs = np.empty(self.n, dtype=object)
            # right_vecs[p] is the part of site>=p and its right that contributes to the marginal of some sites <p
            p = p_given_rightmost
            tens = self.tensors[p][:,state[p]].cuda(gpu)
            tens /= tens.norm()
            right_vecs[p] = (tens@tens.t())
            # right_vecs[p] /= np.trace(right_vecs[p])
            p -= 1
            while p > p_intermid_ungiven_leftmost:
                if givn_mask[p]:
                    tens = self.tensors[p][:,state[p]].cuda(gpu)
                    right_vecs[p] = tens @ right_vecs[p+1] @ tens.t()
                else:
                    tens = self.tensors[p].cuda(gpu)
                    right_vecs[p] = tens.view(-1,tens.shape[-1]) @ right_vecs[p+1]
                    right_vecs[p] = right_vecs[p].view(tens.shape[0],-1) @ (tens.view(tens.shape[0],-1).t())
                # right_vecs[p+1] = right_vecs[p+1].cpu()
                right_vecs[p+1] = right_vecs[p+1].cuda(gpu)
                right_vecs[p] /= right_vecs[p].trace()
                p -= 1
            # print("\nSampling intermediate bits")
            p = p_intermid_ungiven_leftmost
            while p < p_given_rightmost:
                # print('\r %d/%d'%(p,self.n), end="", flush=True)
                tens = self.tensors[p].cuda(gpu)
                if givn_mask[p]==False:
                    # print("Current bit:",p,"given:F","left_vec.shape", left_vec.shape, "matrix.shape", self.tensors[p].shape)
                    right_vec = right_vecs[p+1].cuda(gpu)
                    # prob_marg = tens.view(-1,tens.shape[-1]) @ right_vec
                    # prob_marg = prob_marg.view(tens.shape[0],-1) @ (tens.view(tens.shape[0],-1).t())
                    # prob_marg = (mid_mat_fromleft @ prob_marg @ mid_mat_fromleft.t()).trace()
                    # mid_mat_act = mid_mat_fromleft @ tens[:,1].cuda(gpu)
                    # prob_actv = (mid_mat_act @ right_vec @ mid_mat_act.t()).trace()
                    # if rand()<prob_actv/prob_marg:
                    #     state[p] = 1
                    #     mid_mat_fromleft = mid_mat_act
                    # else:
                    #     state[p] = 0
                    #     mid_mat_fromleft @= tens[:,0].cuda(gpu)
                    mid_mat_1 = mid_mat_fromleft @ tens[:,1].cuda(gpu)
                    mid_mat_0 = mid_mat_fromleft @ tens[:,0].cuda(gpu)
                    prob_1 = (mid_mat_1 @ right_vec @ (mid_mat_1.t())).trace()
                    prob_0 = (mid_mat_0 @ right_vec @ (mid_mat_0.t())).trace()
                    if rand() < prob_1/(prob_0+prob_1):
                        state[p] = 1
                        mid_mat_fromleft = mid_mat_1
                    else:
                        state[p] = 0
                        mid_mat_fromleft = mid_mat_0
                    # givn_mask[p] = True
                else:
                    # print("Current bit:",p,"given:T","left_vec.shape", left_vec.shape, "matrix.shape", self.tensors[p].shape)
                    mid_mat_fromleft @= tens[:,state[p]].cuda(gpu)
                mid_mat_fromleft /= mid_mat_fromleft.norm()
                right_vecs[p+1]=None
                p += 1
            mid_mat_fromleft @= self.tensors[p][:,state[p]].cuda(gpu)
            mid_mat_fromleft /= mid_mat_fromleft.norm()
            p += 1
            # assert givn_mask[p_given_leftmost:p_given_rightmost+1].all()
            # print('')
        for p in range(p_given_rightmost+1, self.n):
            vec_act = mid_mat_fromleft @ self.tensors[p][:,1].cuda(gpu)
            nom = vec_act.norm()
            if rand() < nom**2: #activate
                state[p] = 1
                mid_mat_fromleft = vec_act/nom
            else: #keep 0
                state[p] = 0
                mid_mat_fromleft @= self.tensors[p][:,0].cuda(gpu)
                mid_mat_fromleft /= mid_mat_fromleft.norm()
        for p in np.arange(p_given_leftmost)[::-1]:
            vec_act = self.tensors[p][:,1].cuda(gpu) @ mid_mat_fromleft
            nom = vec_act.norm()
            if rand() < nom**2:
                state[p] = 1
                mid_mat_fromleft = vec_act/nom
            else:
                state[p] = 0
                mid_mat_fromleft = self.tensors[p][:,0].cuda(gpu) @ mid_mat_fromleft
                mid_mat_fromleft /= mid_mat_fromleft.norm()
        assert (state!=-1).all()
        return state
#}}}
#{{{def __getstate__(self):
    def __getstate__(self):
        """ Retuen a dictionary of state values to be pickled """
        exclusion=["tensors","data_cpu","data_test_cpu","data","data_test","idx_gpu","cumulants","gradients","_job_queue","idx_2sites"]
        mydict={}
        for key in self.__dict__.keys():
            if key not in exclusion:
                mydict[key] = self.__dict__[key]
        return mydict
#}}}
#{{{def save(self, fsave_name)
    def save(self, fsave_name=None):
        orig_dir = os.getcwd()
        if fsave_name:
            if fsave_name[0]=='/':
                dit = fsave_name
            else:
                dit = orig_dir+'/'+fsave_name
        else:
            dit = orig_dir+'/L%d'%self.loop
        while os.path.exists(dit):
            dit = dit+'_'
        os.mkdir(dit)
        os.chdir(dit)
        with open('mps.pickle','wb') as fsave:
            tsave=time.time()
            sys.stdout.write("saving mps object to %s ..."%dit);sys.stdout.flush()
            pickle.dump(self,fsave)
            sys.stdout.write("%.2f Sec.\n"%(time.time()-tsave))
        np.savez_compressed('tensors.npz',*[t.numpy() for t in self.tensors])
        if self.m_test>0:
            np.savez_compressed('../data.npz', train=self.data_cpu.numpy().astype(np.uint8), test=self.data_test_cpu.numpy().astype(np.uint8))
        else:
            np.savez_compressed('../data.npz', train=self.data_cpu.numpy().astype(np.uint8), test=None)
        os.chdir(orig_dir)
#}}}

def mpsLoad(path):
    original = os.getcwd()
    os.chdir(path)
    print("Loading",path,end='...',flush=True)
    f = open('mps.pickle','rb')
    mps = pickle.load(f)
    dat = np.load("../data.npz")
    mps.data_cpu=torch.from_numpy(dat['train'].astype(np.int64))
    if dat['test'].size > 1:
        mps.data_test_cpu=torch.from_numpy(dat['test'].astype(np.int64))
    mps.tensors=[]
    tens = np.load('tensors.npz')
    for i in range(mps.n):
        mps.tensors.append(torch.from_numpy(tens['arr_%d'%i]))
    f.close()
    print("Done.")
    os.chdir(original)
    return mps


#{{{if __name__ == "__main__":
if __name__ == "__main__":
    mps = mpsLoad(sys.argv[1])
