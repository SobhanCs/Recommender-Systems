function [biclust] = CF_BPMF( biclust )

hahaha = biclust;
restart = 1;
rand('state',0); 
randn('state',0); 

if restart==1 
  restart=0;
  epsilon=50; % Learning rate 
  lambda  = 0.01; % Regularization parameter 
  momentum=0.8; 

  epoch=1; 
  maxepoch=50; 

%   load moviedata % Triplets: {user_id, movie_id, rating} 

  [num_p num_m] = size(biclust);
  train_vec = makematrix(biclust);
  probe_vec = makematrix0(biclust);
 
  mean_rating = mean(train_vec(:,3)); 
 
  pairs_tr = length(train_vec); % training data 
  pairs_pr = length(probe_vec); % validation data 

  numbatches= 1; % Number of batches  
%   num_m = 3952;  % Number of movies 
%   num_p = 6040;  % Number of users 
  num_feat = 10; % Rank 10 decomposition 

  w1_M1     = 0.1*randn(num_m, num_feat); % Movie feature vectors
  w1_P1     = 0.1*randn(num_p, num_feat); % User feature vecators
  w1_M1_inc = zeros(num_m, num_feat);
  w1_P1_inc = zeros(num_p, num_feat);

end


for epoch = epoch:maxepoch
  rr = randperm(pairs_tr);
  train_vec = train_vec(rr,:);
  clear rr 

  for batch = 1:numbatches
%     fprintf(1,'epoch %d batch %d \r',epoch,batch);
%     N=100000; % number training triplets per batch 
   N=floor(pairs_tr/numbatches); % number training triplets per batch 

    aa_p   = double(train_vec((batch-1)*N+1:batch*N,1));
    aa_m   = double(train_vec((batch-1)*N+1:batch*N,2));
    rating = double(train_vec((batch-1)*N+1:batch*N,3));

    rating = rating-mean_rating; % Default prediction is the mean rating. 

    %%%%%%%%%%%%%% Compute Predictions %%%%%%%%%%%%%%%%%
    pred_out = sum(w1_M1(aa_m,:).*w1_P1(aa_p,:),2);
    f = sum( (pred_out - rating).^2 + ...
        0.5*lambda*( sum( (w1_M1(aa_m,:).^2 + w1_P1(aa_p,:).^2),2)));

    %%%%%%%%%%%%%% Compute Gradients %%%%%%%%%%%%%%%%%%%
    IO = repmat(2*(pred_out - rating),1,num_feat);
    Ix_m=IO.*w1_P1(aa_p,:) + lambda*w1_M1(aa_m,:);
    Ix_p=IO.*w1_M1(aa_m,:) + lambda*w1_P1(aa_p,:);

    dw1_M1 = zeros(num_m,num_feat);
    dw1_P1 = zeros(num_p,num_feat);

    for ii=1:N
      dw1_M1(aa_m(ii),:) =  dw1_M1(aa_m(ii),:) +  Ix_m(ii,:);
      dw1_P1(aa_p(ii),:) =  dw1_P1(aa_p(ii),:) +  Ix_p(ii,:);
    end

    %%%% Update movie and user features %%%%%%%%%%%

    w1_M1_inc = momentum*w1_M1_inc + epsilon*dw1_M1/N;
    w1_M1 =  w1_M1 - w1_M1_inc;

    w1_P1_inc = momentum*w1_P1_inc + epsilon*dw1_P1/N;
    w1_P1 =  w1_P1 - w1_P1_inc;
  end 

  %%%%%%%%%%%%%% Compute Predictions after Paramete Updates %%%%%%%%%%%%%%%%%
  pred_out = sum(w1_M1(aa_m,:).*w1_P1(aa_p,:),2);
  f_s = sum( (pred_out - rating).^2 + ...
        0.5*lambda*( sum( (w1_M1(aa_m,:).^2 + w1_P1(aa_p,:).^2),2)));
  err_train(epoch) = sqrt(f_s/N);

  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  %%% Compute predictions on the validation set %%%%%%%%%%%%%%%%%%%%%% 
  NN=pairs_pr;

  aa_p = double(probe_vec(:,1));
  aa_m = double(probe_vec(:,2));
  rating = double(probe_vec(:,3));

  pred_out = sum(w1_M1(aa_m,:).*w1_P1(aa_p,:),2) + mean_rating;
  ff = find(pred_out>5); pred_out(ff)=5; % Clip predictions 
  ff = find(pred_out<1); pred_out(ff)=1;

  err_valid(epoch) = sqrt(sum((pred_out- rating).^2)/NN);
%   fprintf(1, 'epoch %4i batch %4i Training RMSE %6.4f  Test RMSE %6.4f  \n', ...
%               epoch, batch, err_train(epoch), err_valid(epoch));
  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

  if (rem(epoch,10))==0
     save pmf_weight w1_M1 w1_P1
  end

end 

%   probe_vec(:,3) = pred_out;
%   probe_vec = [probe_vec; train_vec];
%   biclust = zeros(num_p, num_m);
%   for i=1:size(probe_vec, 1)
%       biclust(probe_vec(i,1),probe_vec(i,2)) = probe_vec(i,3);
%   end
% 


%%
restart = 1;
rand('state',0);
randn('state',0);

if restart==1 
  restart=0; 
  epoch=1; 
  maxepoch=50; 

  iter=0; 
%   num_m = 3952;
%   num_p = 6040;
  num_feat = 10;
  
  
  [num_p num_m] = size(hahaha);
  train_vec = makematrix(hahaha);
  probe_vec = makematrix0(hahaha);

  % Initialize hierarchical priors 
  beta=2; % observation noise (precision) 
  mu_u = zeros(num_feat,1);
  mu_m = zeros(num_feat,1);
  alpha_u = eye(num_feat);
  alpha_m = eye(num_feat);  

  % parameters of Inv-Whishart distribution (see paper for details) 
  WI_u = eye(num_feat);
  b0_u = 2;
  df_u = num_feat;
  mu0_u = zeros(num_feat,1);

  WI_m = eye(num_feat);
  b0_m = 2;
  df_m = num_feat;
  mu0_m = zeros(num_feat,1);

%   load moviedata
  mean_rating = mean(train_vec(:,3));
  ratings_test = double(probe_vec(:,3));

  pairs_tr = length(train_vec);
  pairs_pr = length(probe_vec);

  %fprintf(1,'Initializing Bayesian PMF using MAP solution found by PMF \n'); 
%   makematrix
  count = hahaha;


  load pmf_weight
  err_test = cell(maxepoch,1);

  w1_P1_sample = w1_P1; 
  w1_M1_sample = w1_M1; 
  clear w1_P1 w1_M1;

  % Initialization using MAP solution found by PMF. 
  %% Do simple fit
  mu_u = mean(w1_P1_sample)';
  d=num_feat;
  alpha_u = pinv(cov(w1_P1_sample));         %%%%%%%%%%%change inv to pinv

  mu_m = mean(w1_M1_sample)';
  alpha_m = pinv(cov(w1_P1_sample));         %%%%%%%%%%%change inv to pinv

  count=count';
  probe_rat_all = pred(w1_M1_sample,w1_P1_sample,probe_vec,mean_rating);
  counter_prob=1; 

end


for epoch = epoch:maxepoch

  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  %%% Sample from movie hyperparams (see paper for details)  
  N = size(w1_M1_sample,1);
  x_bar = mean(w1_M1_sample)'; 
  S_bar = cov(w1_M1_sample); 

  WI_post = inv(inv(WI_m) + N/1*S_bar + ...
            N*b0_m*(mu0_m - x_bar)*(mu0_m - x_bar)'/(1*(b0_m+N)));
  WI_post = (WI_post + WI_post')/2;

  df_mpost = df_m+N;
  alpha_m = wishrnd(WI_post,df_mpost);
  mu_temp = (b0_m*mu0_m + N*x_bar)/(b0_m+N);  
  lam = chol( inv((b0_m+N)*alpha_m) ); lam=lam';
  mu_m = lam*randn(num_feat,1)+mu_temp;

  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  %%% Sample from user hyperparams
  N = size(w1_P1_sample,1);
  x_bar = mean(w1_P1_sample)';
  S_bar = cov(w1_P1_sample);

  WI_post = inv(inv(WI_u) + N/1*S_bar + ...
            N*b0_u*(mu0_u - x_bar)*(mu0_u - x_bar)'/(1*(b0_u+N)));
  WI_post = (WI_post + WI_post')/2;
  df_mpost = df_u+N;
  alpha_u = wishrnd(WI_post,df_mpost);
  mu_temp = (b0_u*mu0_u + N*x_bar)/(b0_u+N);
  lam = chol( inv((b0_u+N)*alpha_u) ); lam=lam';
  mu_u = lam*randn(num_feat,1)+mu_temp;

  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  % Start doing Gibbs updates over user and 
  % movie feature vectors given hyperparams.  

  for gibbs=1:2 
%     fprintf(1,'\t\t Gibbs sampling %d \r', gibbs);

    %%% Infer posterior distribution over all movie feature vectors 
    count=count';
    for mm=1:num_m
%        fprintf(1,'movie =%d\r',mm);
       ff = find(count(:,mm)>0);
       MM = w1_P1_sample(ff,:);
       rr = count(ff,mm)-mean_rating;
       covar = inv((alpha_m+beta*MM'*MM));
       mean_m = covar * (beta*MM'*rr+alpha_m*mu_m);
       lam = chol(covar); lam=lam'; 
       w1_M1_sample(mm,:) = lam*randn(num_feat,1)+mean_m;
     end

    %%% Infer posterior distribution over all user feature vectors 
     count=count';
     for uu=1:num_p
%        fprintf(1,'user  =%d\r',uu);
       ff = find(count(:,uu)>0);
       MM = w1_M1_sample(ff,:);
       rr = count(ff,uu)-mean_rating;
       covar = inv((alpha_u+beta*MM'*MM));
       mean_u = covar * (beta*MM'*rr+alpha_u*mu_u);
       lam = chol(covar); lam=lam'; 
       w1_P1_sample(uu,:) = lam*randn(num_feat,1)+mean_u;
     end
   end 

   probe_rat = pred(w1_M1_sample,w1_P1_sample,probe_vec,mean_rating);
   probe_rat_all = (counter_prob*probe_rat_all + probe_rat)/(counter_prob+1);
   counter_prob=counter_prob+1;
   
   %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
   %%%%%%% Make predictions on the validation data %%%%%%%
   temp = (ratings_test - probe_rat_all).^2;
   err = sqrt( sum(temp)/pairs_pr);

   iter=iter+1;
   overall_err(iter)=err;

%   fprintf(1, '\nEpoch %d \t Average Test RMSE %6.4f \n', epoch, err);

end 

  probe_vec(:,3) = probe_rat_all;
  probe_vec = [probe_vec; train_vec];
  biclust = zeros(num_p, num_m);
  for i=1:size(probe_vec, 1)
      biclust(probe_vec(i,1),probe_vec(i,2)) = probe_vec(i,3);
  end


end

function [train_vec] = makematrix( biclust )



%% Create a matrix of size num_p by num_m from triplets {user_id, movie_id, rating_id}  

[r c] = find(biclust~=0);
len = length(r);
train_vec = zeros(len,3);
train_vec(:,1) = r;
train_vec(:,2) = c;
train_vec(:,3) = biclust( biclust~=0 );

end 

function [train_vec] = makematrix0( biclust )



%% Create a matrix of size num_p by num_m from triplets {user_id, movie_id, rating_id}  

[r c] = find(biclust==0);
len = length(r);
train_vec = zeros(len,3);
train_vec(:,1) = r;
train_vec(:,2) = c;
train_vec(:,3) = biclust( biclust==0 );

end 
  

function [pred_out] = pred(w1_M1_sample,w1_P1_sample,probe_vec,mean_rating);

%%% Make predicitions on the validation data

 aa_p   = double(probe_vec(:,1));
 aa_m   = double(probe_vec(:,2));
 rating = double(probe_vec(:,3));

 pred_out = sum(w1_M1_sample(aa_m,:).*w1_P1_sample(aa_p,:),2) + mean_rating;
 ff = find(pred_out>5); pred_out(ff)=5;
 ff = find(pred_out<1); pred_out(ff)=1;


 
end