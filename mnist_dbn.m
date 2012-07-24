% mnist_dbn
load('mnist.mat','data');
data = double(data);
ndata = size(data,2);
data = data(:,randperm(ndata));
data = data(:,1:1000);
nv = size(data,1);

epsw = 0.1;
epsbh = 0.1;
epsbv = 0.1;
nepochs = 100;
batchsize = 100;
momentum = 0.5;

% pre-train layer 1
nh1 = 1000;
w1 = 0.1*randn(nh1,nv);
bh1 = zeros(nh1,1);
bv1 = zeros(nv,1);
[w1,bh1,bv1,h1] = train_rbm(w1,bh1,bv1,data,nepochs,batchsize,epsw,epsbh,epsbv,momentum);
save('dbn.mat','w1','bh1','bv1');

% pre-train layer 2
nh2 = 500;
w2 = 0.1*randn(nh2,nh1);
bh2 = zeros(nh2,1);
bv2 = zeros(nh1,1);
[w2,bh2,bv2,h2] = train_rbm(w2,bh2,bv2,h1,nepochs,batchsize,epsw,epsbh,epsbv,momentum);
save('dbn.mat','w1','bh1','bv1','w2','bh2','bv2');

% pre-train layer 3
nh3 = 250;
w3 = 0.1*randn(nh3,nh2);
bh3 = zeros(nh3,1);
bv3 = zeros(nh2,1);
[w3,bh3,bv3,h3] = train_rbm(w3,bh3,bv3,h2,nepochs,batchsize,epsw,epsbh,epsbv,momentum);
save('dbn.mat','w1','bh1','bv1','w2','bh2','bv2','w3','bh3','bv3');

% pre-train layer 4
nh4 = 30;
w4 = 0.1*randn(nh4,nh3);
bh4 = zeros(nh4,1);
bv4 = zeros(nh3,1);
[w4,bh4,bv4,h4] = train_rbm(w4,bh4,bv4,h3,nepochs,batchsize,epsw,epsbh,epsbv,momentum);
save('dbn.mat','w1','bh1','bv1','w2','bh2','bv2','w3','bh3','bv3','w4','bh4','bv4');

% to dream, run alternating gibbs sampling in layer 4 using w4, bh4,
% hb4. take the resulting h3 vectors and instantiate the hidden layer
% of layer 3 with h3. sample down to get h2 using w3 and bh3, etc...
v = data(:,1); % example data input
h1 = sample_up(v,w1,bh1);
h2 = sample_up(h1,w2,bh2);
h3 = sample_up(h2,w3,bh3);
h4 = sample_up(h3,w4,bh4);
for j=1:1000
  %h4 = rand(nh4,1);
  for i=1:10000
    h3 = sample_down(h4,w4,bv4);
    h4 = sample_up(h3,w4,bh4);
  end
  h3 = sample_down(h4,w4,bv4);
  h2 = sample_down(h3,w3,bv3);
  h1 = sample_down(h2,w2,bv2);
  v = sample_down(h1,w1,bv1);
  viewim(v);
  pause(0.03);
end

% to percolate an image up into the higher layers, sample upwards
v = data(:,1); % example data input
h1 = sample_up(v,w1,bh1);
h2 = sample_up(h1,w2,bh2);
h3 = sample_up(h2,w3,bh3);
h4 = sample_up(h3,w4,bh4);

