#!/usr/bin/env python3.6
import torch
import numpy as np
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt

from sys import path, argv
path.append('/home/junwang/')
from mps_mgpu_distSGD import MPS, mpsLoad

store_path = '/data/mnist/28_127.5/rand1k/DmaxGrad/trail2/'
resume_from = '/data/mnist/28_127.5/rand1k/DmaxGrad/trail2/L17D1000_'

def grapher(data):
    m, n_space = data.shape
    aaa = int(n_space**0.5)
    assert n_space == aaa**2
    dat = data.reshape(m, aaa, -1)
    n_row = int(m**0.5)
    dat = np.array_split(dat, n_row)
    n_col = dat[0].shape[0]
    fig, axs = plt.subplots(n_row, n_col, figsize=(n_col, n_row))
    if n_row == 1:
        axs = [axs]
        if n_col == 1:
            axs = [axs]
    for i in range(n_row):
        for j in range(dat[i].shape[0]):
            ax = axs[i][j]
            ax.matshow(dat[i][j]**1.4, cmap="hot_r")
            ax.set_axis_off()
    plt.subplots_adjust(left=0.02, right=0.98, bottom=0.02, top=0.98)


if __name__ == '__main__':
    info = np.load('../../mnist28_127.5.npz')
    dat = torch.from_numpy(info["data"])
    
    Dmax = int(argv[1])
    mps = MPS(dat,1000,[0,1,2,3,4],True,fname="D%d"%Dmax,chunk_size=200)
    mps.max_bondim = Dmax
    mps.grad_mode = "norm0.25"
    mps.learning_rate = 5e-3
    mps.cutoff = 1e-50
    mps.set_batchsize(1000)

    # mps = mpsLoad(resume_from)
    # generate = True
    # Dmax = mps.max_bondim

    # mps.distribute()
    nll_last = mps.nll
    nll_min = nll_last
    tolerator = 0
    generate = False
    for j in range(200):
        mps.train()
        tolerator += 1
        if mps.nll > nll_last:
            mps.learning_rate *= 0.2
        else:
            if mps.nll >= nll_last - 0.002:
                generate = True
                break
            if mps.nll < nll_min:
                mps.save(store_path+"L%dD%d"%(mps.loop,Dmax))
                nll_min = mps.nll
                tolerator = 0
        if tolerator>10:
            break
        nll_last = mps.nll
    mps.save(store_path+"L%dD%d_"%(mps.loop,Dmax))

    if generate:
        mps.left_canonical()
        samples = np.asarray([mps.generate_sample(gid=0) for _ in range(16)])
        grapher(samples)
        np.save('D%dgen.npy'%Dmax,samples)
        plt.savefig('D%dgen.png'%Dmax)

    # np.save(resume_from+'/psis_train.npy',mps.psis.numpy())
    # with open(resume_from+'/psis_train.profile','w') as fp:
    #     print('psi on training samples:\n>0',(mps.psis>0).numpy().sum(),'==0',(mps.psis==0).numpy().sum(),'<0',(mps.psis<0).numpy().sum(),file=fp)
