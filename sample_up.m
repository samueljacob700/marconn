function h = sample_up(v,w,bh)
[nh, nv] = size(w);
h = rand(nh,1) < 1./(1+exp(-w*v-bh));