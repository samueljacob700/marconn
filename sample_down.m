function v = sample_down(h,w,bv)
[nh,nv] = size(w);
v = rand(nv,1) < 1./(1+exp(-w'*h-bv));