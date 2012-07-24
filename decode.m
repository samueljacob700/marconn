function v = decode(h4,w1,w2,w3,w4,bv1,bv2,bv3,bv4);
h3 = sample_down(h4,w4,bv4);
h2 = sample_down(h3,w3,bv3);
h1 = sample_down(h2,w2,bv2);
v = sample_down(h1,w1,bv1);