# Unsupervised Generative Modeling using Matrix Product States

## Class files

* Class `MPS_c` is defined in `MPScumulant.py`.

With a cache for left environments and right environments, it is efficient in DMRG-2.

There's a problem in `numpy.linalg.svd`. In Linux and OS X environments, sometimes we get `numpy.linalg.linalg.LinAlgError: SVD did not converge`, but don't worry, this is rare, only under particular circumstances. On the other hand, if we transfer the problematic matrix to a Windows environment (with Intel MKL), SVD can be carried out. We ascribe this problem to the numerical implementation of SVD in the libraries such as OpenBLAS and LAPACK because mathematically SVD can always be done. **If you have any idea about this issue, any advice will be appreciated!**

## Test files

* In `./BStest` there's an easily repeated experiment, insensitive to most of the hyperparameters.

* `./MNIST` consists data and code for the 1000 images experiment, including training and reconstruction.

## Relevant e-print & Publication

[**Unsupervised Generative Modeling Using Matrix Product States** by *Zhao-Yu Han, Jun Wang, Heng Fan, Lei Wang, Pan Zhang*](https://arxiv.org/abs/1709.01662)