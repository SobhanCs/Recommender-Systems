function [ data] = CF_SLIM( A)

[N,M] = size(A);
lambda = 1;
beta = 5;
opts=[];
opts.rsL2=beta;
% opts.maxIter=200;
% opts.tol=0.001;

W=[];
for j=1:M
    %j
    %tmp_T = tic;
    [wj, funVal] = nnLeastR(A, A(:,j), lambda, opts);
    %elapse = toc(tmp_T)
    W = [W wj];
end

data = A*W;

return