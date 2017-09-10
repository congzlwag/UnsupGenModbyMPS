# ./

* _data.npy saves a (1000, 28*28) numpy array which is the binary data;

* _prob.npy saves a (1000,) numpy array which is the empirical probability distribution of the dataset in _data.npy;

* _shannon.npy saves the Shannon entropy of the dataset in _data.npy;

* example.pdf shows some of the images in the dataset;

* indices.npy indicate the indices of these 1000 images in the original MNIST database.

## ./recon-denos

* ori_indices.npy saves the indices among the 1000 images for the 20 images in original.pdf;

* annoised05.npy, annoised10.npy and annoised20.npy are the results of adding different levels of noise (5%, 10%, 20%) on these 20 images, respectively.

* otherindx.npy saves 20 indices different from those in ori_indices.npy. These 20 indices are also picked from range(1000), but they are excluded from ori_indices.npy; otherimg.eps shows them.