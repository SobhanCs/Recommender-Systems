function [ pre ] = CF_PPMF( train )
%CF_PPMF Summary of this function goes here
%   Detailed explanation goes here

[N,M]=size(train);
R_train = sparse(train);
R_v = R_train;
R_test = R_train;

mask_train = zeros(N,M);
for i=1:N
    for j=1:M
        if (train(i,j)~=0)
            mask_train(i,j) = 1;
        end
    end
end
mask_v = mask_train;
mask_test = ones(N,M);

[N,M]=size(R_train);

k=5;

% set the init type
inittype=2;


% perform ppmf given the initial value for variational parameters
if inittype==2;
    init.Lambda1=rand(k,N);
    init.Nu1=rand(k,N);
    init.Lambda2=rand(k,M);
    init.Nu2=rand(k,M);
    [mu1,Sigma1,mu2,Sigma2,tau,Lambda1,Nu1,Lambda2,Nu2,mv]=ppmfLearn(R_train,mask_train,R_v,mask_v,inittype,init);
    % prediction on the whole matrix
    [R_pred,rmse]=ppmfPred(Lambda1,Lambda2,mv,R_test,mask_test);
% perform ppmf given the initial value for model parameters
elseif inittype==1
    init.mu1=rand(k,1);
    init.Sigma1=rand(k,k);
    init.mu2=rand(k,1);
    init.Sigma2=rand(k,k);
    init.tau=1;
    [mu1,Sigma1,mu2,Sigma2,tau,Lambda1,Nu1,Lambda2,Nu2,mv]=ppmfLearn(R_train,mask_train,R_v,mask_v,inittype,init);
    % prediction on the whole matrix
    [R_pred,rmse]=ppmfPred(Lambda1,Lambda2,mv,R_test,mask_test);
end

pre = full(R_pred);

end



function [Lambda1_t,Nu1_t,Lambda2_t,Nu2_t]=ppmfEstep(R,mask,mu1,Sigma1,mu2,Sigma2,tau,Lambda1,Nu1,Lambda2,Nu2)
% 
% Author: Hanhuai Shan. 04/2012.  
%
% E-step of ppmf
%
%   k:          the rank after decomposition
%   M:          #movies
%   N:          #users
%
% Input:
%   --data--
%   R:          N*M, rating matrix for learning
%   mask:       N*M, indicator matrix for R, 1 denotes non-missing entry, and 0 denotes missing entry
%   steps:      steps for the E step
%   --model parameters--
%   mu1:        k*1
%   Sigma1:     k*k
%   mu2:        k*1
%   Sigma2:     k*k
%   tau:        scaler
%   --initializations for variational parameters in the current E step--
%   Lambda1:    k*N
%   Nu1:        k*M
%   Lambda2:    k*N
%   Nu2:        k*M
%
% Output:
%   Lambda1_t:  k*N
%   Nu1_t:      k*M
%   Lambda2_t:  k*N
%   Nu2_t:      k*M
%---------------------------------------------------------------------


k=length(mu1);
[N,M]=size(R);

Lambda1_t=Lambda1;
Nu1_t=Nu1;
Lambda2_t=Lambda2;
Nu2_t=Nu2;

invSigma1=inv(Sigma1);
invSigma2=inv(Sigma2);
invSigma1mu1=invSigma1*mu1;
invSigma2mu2=invSigma2*mu2;


t=1;
steps=10;

% We run the iterations for fixed times of steps.
% Alternatives would be tracking the change of log-likelihood, or the
% change of the variational parameters, etc..
while t<steps
    
    %update Lambda1
    for j=1:M
        Lambda2_t_square_temp(:,:,j)=Lambda2_t(:,j)*Lambda2_t(:,j)'/tau;
    end
    Nu2_t_temp=Nu2_t*mask'/tau;
    right_temp=invSigma1mu1*ones(1,N)+Lambda2_t*(R.*mask)'/tau;
    
    
    for i=1:N
        J=find(mask(i,:)==1);
        temp=sum(Lambda2_t_square_temp(:,:,J),3);

        left=invSigma1+temp+diag(Nu2_t_temp(:,i));
        right=right_temp(:,i);
        Lambda1_tt(:,i)=inv(left)*right;   
    end
    
    % Update Lambda2
    for i=1:N
        Lambda1_tt_square_temp(:,:,i)=Lambda1_tt(:,i)*Lambda1_tt(:,i)'/tau;
    end   
    Nu1_t_temp=Nu1_t*mask/tau;
    right_temp=(invSigma2mu2)*ones(1,M)+Lambda1_tt*(R.*mask)/tau;
 
    for j=1:M
        I=find(mask(:,j)==1);
        temp=sum(Lambda1_tt_square_temp(:,:,I),3);
        left=invSigma2+temp+diag(Nu1_t_temp(:,j));
        right=right_temp(:,j);
        Lambda2_tt(:,j)=inv(left)*right;
    end
    
    % Update Nu1
    Nu1_tt=1./((Lambda2_tt.^2+Nu2_t)*mask'/tau+diag(invSigma1)*ones(1,N));
    
    % Update Nu2
    Nu2_tt=1./((Lambda1_tt.^2+Nu1_tt)*mask/tau+diag(invSigma2)*ones(1,M));
   
    % set up for the next iteration
    Lambda1_t=Lambda1_tt;
    Lambda2_t=Lambda2_tt;
    Nu1_t=Nu1_tt;
    Nu2_t=Nu2_tt;
    
    t=t+1;
   
end

end


function [mu1,Sigma1,mu2,Sigma2,tau]=ppmfMstep(R,mask,Lambda1,Nu1,Lambda2,Nu2) 
% 
% Author: Hanhuai Shan. 04/2012.  
%
% M step of ppmf
%   
%   k:          the rank after decomposition
%   M:          #movies
%   N:          #users
%
% Input:
%   R:          N*M, rating matrix for learning
%   mask:       N*M, indicator matrix for R, 1 is non-missing entry
%   Lambda1:    k*N
%   Nu1:        k*M
%   Lambda2:    k*N
%   Nu2:        k*M
%
% Output:
%   mu1:        k*1
%   Sigma1:     k*k
%   mu2:        k*1
%   Sigma2:     k*k
%   tau:        scaler
%-----------------------------------------------------------------

[N,M]=size(R);
[k,N]=size(Lambda1);


% Update mu1
mu1=sum(Lambda1,2)/N;

% Update mu2
mu2=sum(Lambda2,2)/M;

% Sigma1
temp=0;
for i=1:N
    temp=temp+diag(Nu1(:,i))+(Lambda1(:,i)-mu1)*(Lambda1(:,i)-mu1)';
end
Sigma1=temp/N;
Sigma1=Sigma1+exp(-30);

% Sigma2
temp=0;
for j=1:M
    temp=temp+diag(Nu2(:,j))+(Lambda2(:,j)-mu2)*(Lambda2(:,j)-mu2)';
end
Sigma2=temp/M;
Sigma2=Sigma2+exp(-30);

%tau
tau=sum(sum((R.^2-2*R.*(Lambda1'*Lambda2)...
    +(Lambda1'*Lambda2).^2 ...
    +Lambda1'.^2*Nu2+Nu1'*(Lambda2.^2)+Nu1'*Nu2).*mask))/sum(sum(mask));

end

function [R_pred,rmse]=ppmfPred(U,V,mv,R_test,mask_test)
% 
% Author: Hanhuai Shan. 04/2012.  
% 
% Predict on the test set
%
%   k:          the rank after decomposition
%   M:          #movies
%   N:          #users
%
% Input:
%   --- factorization results from the learning process---
%   U:    k*N
%   V:    k*M
%   mv:   mean of all non-missing entries in R_train
%   ---data---
%   R_test:          N*M, rating matrix for test
%   mask_test:       N*M, indicator matrix for R_test, 1 is non-missing entry
%
%
% Output:
%   R_pred:         Result for prediciton
%   rmse:           RMSE
%----------------------------------------------------------------------

R_pred=(U'*V+mv).*mask_test;
rmse=sqrt(sum(sum((R_pred-R_test).^2.*mask_test))/sum(sum(mask_test)));

end

function [mu1,Sigma1,mu2,Sigma2,tau,Lambda1_t,Nu1_t,Lambda2_t,Nu2_t,mv]=ppmfLearn(R_train,mask_train,Rv,maskv,inittype,init)
% 
% Author: Hanhuai Shan. 04/2012.  
%
% Learn ppmf by runing variational EM
%
%   k:          the rank after decomposition
%   M:          #movies
%   N:          #users
%
% Input:
%   ---data---
%   R_train:    N*M, rating matrix for training
%   mask_train: N*M, indicator matrix for R_train, 1 is non-missing entry
%   Rv:         N*M, rating matrix for valdation to determin early stopping
%   maskv:      N*M, indicator matrix for RV
%   ---initialization---
%   inittype:   if inittype==1, model parameters are initialized; 
%               if inittype==2, variational parameters are initialized;
%   init:    
%   if inittype ==1, then init contains:
%   ---initialization for model parameters---
%   mu1:        k*1
%   Sigma1:     k*k
%   mu2:        k*1
%   Sigma2:     k*k
%   tau:        scaler
%   if inittype==2, then init contains:
%   --- initialization for variational parameters---
%   Lambda1:    k*N
%   Nu1:        k*N
%   Lambda2:    k*M
%   Nu2:        k*M
%
% Output:
%   mu1_t:        k*1
%   Sigma1_t:     k*k
%   mu2_t:        k*1
%   Sigma2_t:     k*k
%   tau_t:        scaler
%   Lambda1:      k*N
%   Nu1:          k*M
%   Lambda2:      k*N
%   Nu2:          k*M
%   mv:           mean of all non-missing entries in R_train
%----------------------------------------------------------------------


[N,M]=size(R_train);
mv=sum(sum(R_train.*mask_train))/sum(sum(mask_train)); % mean of all non-missing entries
R_train=(R_train-mv).*mask_train;

epsilon=0.001;
steps=500;
t=1;
valError_t=100;
minstep=5;

% if only given the model parameters, randomly initialize the variational
% parameters
if inittype==1
    mu1=init.mu1;
    Sigma1=init.Sigma1;
    mu2=init.mu2;
    Sigma2=init.Sigma2;
    tau=init.tau;
    k=length(mu1);
    Lambda1_t=0.1*rand(k,N);
    Nu1_t=rand(k,N);
    Lambda2_t=0.1*rand(k,M);
    Nu2_t=rand(k,M);
% if only given the variational parameters, perform an E-step first to get
% the model parameters
elseif inittype==2
    Lambda1_t=init.Lambda1;
    Lambda2_t=init.Lambda2;
    Nu1_t=init.Nu1;
    Nu2_t=init.Nu2;
    [mu1,Sigma1,mu2,Sigma2,tau]=ppmfMstep(R_train,mask_train,Lambda1_t,Nu1_t,Lambda2_t,Nu2_t);
    k=length(mu1);
else
    disp('input error.')
end

while t<steps %&& e>epsilon 
    % E-step
    [Lambda1_tt,Nu1_tt,Lambda2_tt,Nu2_tt]=ppmfEstep(R_train,mask_train,mu1,Sigma1,mu2,Sigma2,tau,Lambda1_t,Nu1_t,Lambda2_t,Nu2_t);
     
    % M-step   
    [mu1,Sigma1,mu2,Sigma2,tau]=ppmfMstep(R_train,mask_train,Lambda1_tt,Nu1_tt,Lambda2_tt,Nu2_tt);
        
    % error for validation
    valError_tt=sqrt(sum(sum((Lambda1_tt'*Lambda2_tt+mv-Rv).^2.*maskv))/sum(sum(maskv)));
    e=(valError_t-valError_tt)/valError_t; % difference of validation error from the last iteration
%     disp(['t=',int2str(t),' rmse= ',num2str(full(valError_tt))]);    
    
    % setup for the next iteration
    Lambda1_t=Lambda1_tt;
    Lambda2_t=Lambda2_tt;
    Nu1_t=Nu1_tt;
    Nu2_t=Nu2_tt;
    valError_t=valError_tt;
    t=t+1;
    
    % After minstep iterations, if the validation error stops decreasing in
    % any iteration, finish.
    if e<epsilon && t>=minstep
        break;
    end
    
end

end
