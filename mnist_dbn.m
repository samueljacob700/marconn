% mnist_dbn
load('mnist.mat','data');
data = double(data);
nv = size(data,1);
ndata = size(data,2);

epsw = 0.1;
epsbh = 0.1;
epsbv = 0.1;
nepochs = 100;
batchsize = 100;

nh = 100;
w = 0.1*randn(nh,nv);
bh = zeros(nh,1);
bv = zeros(nv,1);

% [w,bh,bv] = train_rbm(w,bh,bv,data,nepochs,batchsize,epsw,epsbh,epsbv)
[w,bh,bv] = train_rbm(w,bh,bv,data,nepochs,batchsize,epsw,epsbh,epsbv);


