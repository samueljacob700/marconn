% mnist_dbn
load('mnist_all.mat');
data = [test0;test1;test2;test3;test4;test5;test6;test7;test8;test9;...
        train0;train1;train2;train3;train4;train5;train6;train7;train8;train9];
data = double(data')/255;
data = double(data > 0.5); % all binary input
ndata = size(data,2);
data = data(:,randperm(ndata));
nv = size(data,1);

