function vout = sample_rbm(w,bh,bv,vinit,ns)

[nh, nv] = size(w);
vout = zeros(nv,ns);

v = vinit;
for s=1:ns
    ph = 1./(1+exp(-(bh + w*v)));
    h = rand(nh,1) < ph;
%     % add some noise to h
%     change = rand(nh,1) < 0.1;
%     h(change) = ~h(change);
    pv = 1./(1+exp(-(bv + w'*h)));
    v = rand(nv,1) < pv;
    vout(:,s) = v;
    viewim(v);
    pause(0.001);
end