% just learn a single-layer RBM
load('mnist_all.mat');
data = [test0;test1;test2;test3;test4;test5;test6;test7;test8;test9;...
        train0;train1;train2;train3;train4;train5;train6;train7;train8;train9];
data = double(data')/255;
%data = double(data > 0.5); % all binary input
ndata = size(data,2);
data = data(:,randperm(ndata));

params.epsw = 0.1;
params.epsbh = 0.1;
params.epsbv = 0.1;
params.nepochs = 500;
params.batchsize = 100;
params.initialmomentum = 0.5;
params.finalmomentum = 0.9;
params.weightcost = 0.0002;

nv = size(data,1);
nh1 = 1000;
w1 = 0.1*randn(nh1,nv);
bh1 = zeros(nh1,1);
bv1 = zeros(nv,1);
[w1,bh1,bv1] = train_rbm_cd1(w1,bh1,bv1,data,params);
save('rbm.cd1.1000.500','w1','bh1','bv1','params');
%sample_rbm(w1,bh1,bv1,data(:,1),1000);
