function [w,bh,bv] = train_rbm(w,bh,bv,data,params)
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
    p_h_v = 1./(1+exp(bsxfun(@plus,-w*batchdata,-bh))); % nh x batchsize % B
    Ew_data = (1/batchsize)*p_h_v*batchdata'; % nh x nv
    Ebh_data = (1/batchsize)*sum(p_h_v,2); % nh x 1
    Ebv_data = (1/batchsize)*sum(batchdata,2); % nv x 1
    
    if strcmp(params.mode,'CDk')
      % E_model (use CD-k)
      % re-use the following computations from the E_data computation
      % v = data;
      % p_h_v = 1./(1+exp(bsxfun(@plus,-w*v,-bh))); % nh x batchsize % neghidprobs
      h = double(rand(nh,batchsize) < p_h_v); % always sample here
      for i=1:params.CDk
	p_v_h = 1./(1+exp(bsxfun(@plus,-w'*h,-bv))); % nv x batchsize % negdata	
	if params.use_vis_probs
	  v = p_v_h;
	else
	  v = double(rand(nv,batchsize) < p_v_h); 	
	end
	p_h_v = 1./(1+exp(bsxfun(@plus,-w*v,-bh))); % nh x batchsize % neghidprobs
	if i == params.CDk % on last iteration, use probabilities, do not sample
	  h = p_h_v;
	else
	  h = double(rand(nh,batchsize) < p_h_v);
	end
      end
      Ew_model = (1/batchsize)*(h*v'); % negprods
      Ebh_model = (1/batchsize)*sum(h,2);
      Ebv_model = (1/batchsize)*sum(v,2);
    elseif strcmp(params.mode,'ML_grad')
      % E_model (use ML gradient -- Gibbs sampling with random initialization)
      v = double(rand(nv,1) < 0.5);
      v_samples = zeros(nv,nGibbs-Gibbs_burnin);
      h_samples = zeros(nh,nGibbs-Gibbs_burnin);
      for i=1:params.nGibbs
	if i > params.Gibbs_burnin % after burn-in period
	  p_h_v = 1./(1+exp(bsxfun(@plus,-w*v,-bh))); % nh x 1
	  h = double(rand(nh,1) < p_h_v);
	  p_v_h = 1./(1+exp(bsxfun(@plus,-w'*h,-bv))); % nv x batchsize % negdata	
	  v = double(rand(nv,1) < p_v_h);
	  v_samples(:,i - params.Gibbs_burnin) = v;
	  h_samples(:,i - params.Gibbs_burnin) = h;
	end
      end
      Ew_model = (1/(nGibbs-Gibbs_burnin))*h_samples*v_samples';
      Ebh_model = (1/(nGibbs-Gibbs_burnin))*sum(h_samples,2);
      Ebv_model = (1/(nGibbs-Gibbs_burnin))*sum(v_samples,2);
    elseif strcmp(params.mode,'PCD')
      % E_model (use Persistent Contrastive Divergence: Tieleman
      % Training Restricted Boltzmann Machines using Approximations to
      % the Likelihood Gradient)
      

    end

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