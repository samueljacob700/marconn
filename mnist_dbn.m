% mnist_dbn
load('mnist_all.mat');
data = [test0;test1;test2;test3;test4;test5;test6;test7;test8;test9;...
        train0;train1;train2;train3;train4;train5;train6;train7;train8;train9];
data = double(data')/255;
data = double(data > 0.5); % all binary input
ndata = size(data,2);
data = data(:,randperm(ndata));
nv = size(data,1);

params.epsw = 0.1;
params.epsbh = 0.1;
params.epsbv = 0.1;
params.nepochs = 200;
params.batchsize = 100;
params.initialmomentum = 0.5;
params.finalmomentum = 0.5;
params.weightcost = 0.0002;

% pre-train layer 1
nh1 = 1000;
w1 = 0.1*randn(nh1,nv);
bh1 = zeros(nh1,1);
bv1 = zeros(nv,1);
[w1,bh1,bv1,h1] = train_rbm(w1,bh1,bv1,data,params);
save('dbn.mat','w1','bh1','bv1');

% pre-train layer 2
nh2 = 500;
w2 = 0.1*randn(nh2,nh1);
bh2 = zeros(nh2,1);
bv2 = zeros(nh1,1);
[w2,bh2,bv2,h2] = train_rbm(w2,bh2,bv2,h1,params);
save('dbn.mat','w1','bh1','bv1','w2','bh2','bv2');

% pre-train layer 3
nh3 = 250;
w3 = 0.1*randn(nh3,nh2);
bh3 = zeros(nh3,1);
bv3 = zeros(nh2,1);
[w3,bh3,bv3,h3] = train_rbm(w3,bh3,bv3,h2,params);
save('dbn.mat','w1','bh1','bv1','w2','bh2','bv2','w3','bh3','bv3');

% pre-train layer 4
nh4 = 30;
w4 = 0.1*randn(nh4,nh3);
bh4 = zeros(nh4,1);
bv4 = zeros(nh3,1);
[w4,bh4,bv4,h4] = train_rbm(w4,bh4,bv4,h3,params);
save('dbn.mat','w1','bh1','bv1','w2','bh2','bv2','w3','bh3','bv3','w4','bh4','bv4');

% to dream, run alternating gibbs sampling in layer 4 using w4, bh4,
% hb4. take the resulting h3 vectors and instantiate the hidden layer
% of layer 3 with h3. sample down to get h2 using w3 and bh3, etc...

dream(w4,bh4,bv4,w3,bh3,bv3,w2,bh2,bv2,w1,bh1,bv1);

%h4 = encode(data(:,1),w1,w2,w3,w4,bh1,bh2,bh3,bh4);
%v = decode(h4,w1,w2,w3,w4,bv1,bv2,bv3,bv4);
