function [w,bh,bv] = train_rbm_cd1(w,bh,bv,data,params)
% train an RBM with binary visible and hidden units using contrastive divergence (CD-K)
% hout contains a sample from the final model for each data example
nepochs = params.nepochs;
batchsize = params.batchsize;
epsw = params.epsw;
epsbh = params.epsbh;
epsbv = params.epsbv;
initialmomentum = params.initialmomentum;
finalmomentum = params.finalmomentum;
weightcost = params.weightcost;

[nv,ndata] = size(data);
nh = size(bh,1);

assert(rem(ndata,batchsize) == 0);
nbatch = ndata/batchsize;

d_w = zeros(nh,nv);
d_bh = zeros(nh,1);
d_bv = zeros(nv,1);

for epoch=1:nepochs
  fprintf(1,'epoch %d\r',epoch); 
  errsum = 0;

  for batch=1:nbatch
    first = (batch-1)*batchsize+1;
    last = batch*batchsize;
    batchdata = data(:,first:last); % nv x batchsize % A
    
    % E_data (exact)
    p_h_data = 1./(1+exp(bsxfun(@plus,-w*batchdata,-bh))); % nh x batchsize % B
    Ew_data = (1/batchsize)*p_h_data*batchdata'; % nh x nv
    Ebh_data = (1/batchsize)*sum(p_h_data,2); % nh x 1
    Ebv_data = (1/batchsize)*sum(batchdata,2); % nv x 1
    
    % E_model (use CD-1 approximation)
    % using some approximations and tricks from Hinton's code and paper
    % initialize Gibbs chain from data
    h = double(rand(nh,batchsize) < p_h_data); % nh x batchsize % poshidstates
    p_v_h = 1./(1+exp(bsxfun(@plus,-w'*h,-bv))); % nv x batchsize % negdata
    p_h_v = 1./(1+exp(bsxfun(@plus,-w*p_v_h,-bh))); % nh x batchsize % neghidprobs
    Ew_model = (1/batchsize)*(p_h_v*p_v_h'); % negprods
    Ebh_model = (1/batchsize)*sum(p_h_v,2);
    Ebv_model = (1/batchsize)*sum(p_v_h,2);

    err = sum(sum( (batchdata-p_v_h).^2 ));
    errsum = err + errsum;    
    if epoch > 5
      momentum = finalmomentum;
    else
      momentum = initialmomentum;
    end
    
    % update
    d_w = momentum*d_w + epsw*(Ew_data - Ew_model) - weightcost*w;
    d_bh = momentum*d_bh + epsbh*(Ebh_data - Ebh_model);
    d_bv = momentum*d_bv + epsbv*(Ebv_data - Ebv_model);
    w = w + d_w;
    bh = bh + d_bh;
    bv = bv + d_bv;
    
    
  end
  fprintf(1, 'epoch %4i error %6.1f  \n', epoch, errsum); 
end