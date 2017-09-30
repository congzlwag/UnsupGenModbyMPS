rng(1)
clear mps;
n=784;
m=100;
Dmax=20; % maximum bond dimension
n_batches=1; % number of batches in the stochastic gradient descent
load('mnist_100_images.mat'); % 100 mnist training images that have been randomly selected from 60000 training images
mps=MPS(n,train_x_binary,n_batches);
mps.max_bondim=Dmax;
mps.learning_rate=0.001;
mps.train(20); % training for 20 loops
s=mps.generate_sample(100)-1; %generate new samples


