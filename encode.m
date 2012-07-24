function h4 = encode(v,w1,w2,w3,w4,bh1,bh2,bh3,bh4);
h1 = sample_up(v,w1,bh1);
h2 = sample_up(h1,w2,bh2);
h3 = sample_up(h2,w3,bh3);
h4 = sample_up(h3,w4,bh4);