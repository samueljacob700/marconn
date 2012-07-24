function dream(w4,bv4,w3,bv3,w2,bv2,w1,bv1)

for j=1:1000
  h4 = rand(nh4,1);
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