function dream(w4,bh4,bv4,w3,bh3,bv3,w2,bh2,bv2,w1,bh1,bv1,v)
 nh4 = size(w4,1);
 
 if nargin == 13
   % imagination based on input
   while(1)
     h1 = sample_up(v,w1,bh1);
     h2 = sample_up(h1,w2,bh2);
     h3 = sample_up(h2,w3,bh3);
     h4 = sample_up(h3,w4,bh4);
     h3 = sample_down(h4,w4,bv4);
     h2 = sample_down(h3,w3,bv3);
     h1 = sample_down(h2,w2,bv2);
     v1 = sample_down(h1,w1,bv1);
     viewim(v1);
     pause(0.01);
   end

   
 else
   % dreaming
   h4 = rand(nh4,1);
   while(1)
     % randomly change
     %change = rand(nh4,1) < 0.05;
     %h4(change) = ~h4(change);
     for i=1:1000
       h3 = sample_down(h4,w4,bv4);       
       h4 = sample_up(h3,w4,bh4);
     end
     h2 = sample_down(h3,w3,bv3);
     h1 = sample_down(h2,w2,bv2);
     v = sample_down(h1,w1,bv1);
     
     viewim(v);
     pause(0.001);
   end
   
 end