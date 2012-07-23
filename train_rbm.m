function [w,bh,bv] = train_rbm(w,bh,bv,data,nepochs,batchsize,epsw,epsbh,epsbv)

[nv,ndata] = size(data);
nh = size(bh,1);
CDk = 1;

assert(rem(ndata,batchsize) == 0);
nbatch = ndata/batchsize;

for epoch=1:nepochs
    
    for batch=1:nbatch
        first = (batch-1)*batchsize+1;
        last = batch*batchsize;
        batchdata = double(data(:,first:last)); % nv x batchsize
        
        % E_data (exact)
        p_h_data = 1./(1+exp(bsxfun(@plus,-w*batchdata,-bh))); % nh x batchsize
        Ew_data = (1/batchsize)*p_h_data*batchdata'; % nh x nv
        Ebh_data = (1/batchsize)*sum(p_h_data,2); % nh x 1
        Ebv_data = (1/batchsize)*sum(batchdata,2); % nv x 1
        
        % E_model (use CD-K approximation)
        % initialize Gibbs chain from data
        h = double(rand(nh,batchsize) < p_h_data); % batchsize x nh
        for k=1:CDk
            % sample visible given hidden
            tmp = -w'*h;
            p_v_h = 1./(1+exp(bsxfun(@plus,tmp,-bv))); % batchsize x nv
            v = double(rand(nv,batchsize) < p_v_h); % batchsize x nv
            % sample hidden given visible TODO use probabilities--mean field(?)
            tmp = -w*v;
            p_h_v = 1./(1+exp(bsxfun(@plus,tmp,-bh))); % batchsize x nh (TODO reuse above)
            h = double(rand(nh,batchsize) < p_h_v); % batchsize x nh
        end
        Ew_model = (1/batchsize)*(h*v');
        Ebh_model = (1/batchsize)*sum(h,2);
        Ebv_model = (1/batchsize)*sum(v,2);
        
        % update
        w = w + epsw*(Ew_data - Ew_model);
        bh = bh + epsbh*(Ebh_data - Ebh_model);
        bv = bv + epsbv*(Ebv_data - Ebv_model);
        
    end
end