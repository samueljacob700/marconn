function [w,bh,bv,hout] = train_rbm(w,bh,bv,data,nepochs,batchsize,epsw,epsbh,epsbv,momentum)
% hout contains a sample from the final model for each data example

[nv,ndata] = size(data);
nh = size(bh,1);
CDk = 1;

assert(rem(ndata,batchsize) == 0);
nbatch = ndata/batchsize;

d_w = zeros(nh,nv);
d_bh = zeros(nh,1);
d_bv = zeros(nv,1);

for epoch=1:nepochs
    epoch
    for batch=1:nbatch
        first = (batch-1)*batchsize+1;
        last = batch*batchsize;
        batchdata = data(:,first:last); % nv x batchsize
        
        % E_data (exact)
        p_h_data = 1./(1+exp(bsxfun(@plus,-w*batchdata,-bh))); % nh x batchsize
        Ew_data = (1/batchsize)*p_h_data*batchdata'; % nh x nv
        Ebh_data = (1/batchsize)*sum(p_h_data,2); % nh x 1
        Ebv_data = (1/batchsize)*sum(batchdata,2); % nv x 1
        
        % E_model (use CD-K approximation)
        % initialize Gibbs chain from data
        h = double(rand(nh,batchsize) < p_h_data); % nh x batchsize
	hout(:,first:last) = h;
        for k=1:CDk
            % sample visible given hidden
            tmp = -w'*h;
            p_v_h = 1./(1+exp(bsxfun(@plus,tmp,-bv))); % nv x batchsize
            v = double(rand(nv,batchsize) < p_v_h);
            % sample hidden given visible TODO use probabilities--mean field(?)
            tmp = -w*v;
            p_h_v = 1./(1+exp(bsxfun(@plus,tmp,-bh))); % nh x batchsize
            h = double(rand(nh,batchsize) < p_h_v);
        end
        Ew_model = (1/batchsize)*(h*v');
        Ebh_model = (1/batchsize)*sum(h,2);
        Ebv_model = (1/batchsize)*sum(v,2);

        % update
	d_w = momentum*d_w + epsw*(Ew_data - Ew_model);
	d_bh = momentum*d_bh + epsbh*(Ebh_data - Ebh_model);
	d_bv = momentum*d_bv + epsbv*(Ebv_data - Ebv_model);
        w = w + d_w;
        bh = bh + d_bh;
        bv = bv + d_bv;
        
    end
end