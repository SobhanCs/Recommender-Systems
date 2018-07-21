clear; close all;

%% Loading the Data
M = dlmread('ml-100k/u.data');  %1
S = spconvert(M(:,1:3));          %Sparse
mmwrite('u.mtx', S);              %MatrixMarket %2

%% Make Directory to save data 
if( ~exist('results', 'dir') )
    mkdir('results');
end
if( ~exist('testbed', 'dir') )
    mkdir('testbed');
end

%% Setting the Dimension
if strcmp(dataset,'movielens-100k')
        dim = 6;
    else
        dim = 3;  % # of dimensions reduced to
end

%% Preprocessing Data
ratio_train = 0.8;
rand( 'twister' ,5489);  %use rng instead  %RNG:control random number of generation
dataset_list = {'movielens-100k_nmf_merge5'};   %List
dataset = cell2mat(dataset_list(1));    %ConvertCellToMatrix
datafile = strcat(dataset,'_train',num2str(ratio_train),'_dim',num2str(dim));  %Concatenation
datafile_cf = strcat(dataset,'_train',num2str(ratio_train));

%% Converting Data to Double
if(~exist('data','var'))
    if(exist('S','var'))
        data = double(S);
        clear S;
    elseif(exist('rating','var'))
        data = double(rating);
        clear rating;
    elseif(exist('u_train','var'))
        data = double(u_train);
        clear u_train;
    end
end

ClsNum = 50;
for ClassNum =2:ClsNum

%% Split the train and test parts
[N,M] = size(data);
userCount =sum(data>0,2);
trainCount = round(userCount * ratio_train);
u_train = zeros(size(data));
u_test = u_train;
for u = 1:N
    idx = randperm(userCount(u));
    u_rates = data(u,:)>0;
    tmp = data(u,u_rates); %train
    tmp2 = tmp; %test
    tmp(idx(trainCount(u)+1:end)) = 0;
    tmp2(idx(1:trainCount(u))) = 0;
    u_train(u,u_rates) = tmp;
    u_test(u,u_rates) = tmp2;    
end
u_train = sparse(u_train);
u_test = sparse(u_test);

%% STEP1 : dimensionality reduction

%tic;
%opt = statset('Maxiter',6);
%T = hosvd( u_train , ClassNum );
opt = statset('Maxiter',500,'Display','final'); %6
[VM,UM] = nnmf(u_train,ClassNum,'options',opt);
%[VM,UM] = nmfpark(u_train,ClassNum,'type','plain','nnls_solver','as');
re = norm(u_train-VM*UM,'fro')/norm(u_train,'fro');
xlswrite(sprintf('re%.0f',ClassNum),re)
%[VM,UM] = seminmf(u_train,ClassNum);
%[VM,UM] = nnmf(u_train,ClsNum,'options',opt);
%[VM,UM] = deep_seminmf(u_train, ClsNum);
%[VM,UM] = nmf(u_train, ClsNum);
%[VM,UM] = GNMF(u_train, ClsNum);
Vdata=[VM;UM'];
lm=sum(Vdata')>0;
l1=find(lm==0);
Vdata(l1,:)=1/ClassNum;
%t1=toc;

%[Vdata] = DimReduce(u_train,dim); %3   %4 - constructW   %5 - Eudist2

%% Saving Results
file = strcat('testbed\',dataset,'_train',num2str(ratio_train),'_dim',num2str(dim),'.mat');  %Concatenation
save(file, 'u_train','u_test','ratio_train','Vdata');  %Save
load(strcat('testbed\',datafile,'.mat'));
u_train = full(u_train);
if(~exist(strcat('testbed\',datafile,'.mat'),'file') )
        prepare_dataset(dataset,ratio_train,dim);
end
    load(strcat('testbed\',datafile,'.mat'));
    
u_train = full(u_train);
%fprintf('======================\n');
%fprintf('* Base Method\n');
% Base Method
%runCF(datafile_cf, u_train, u_test, 'POP' );
%runCF(datafile_cf, u_train, u_test, 'UserBased' );
%runCF(datafile_cf, u_train, u_test, 'PMF' );
%runCF(datafile_cf, u_train, u_test, 'BPMF' );
%runCF(datafile_cf, u_train, u_test, 'PPMF' );
%runCF(datafile_cf, u_train, u_test, 'SVD' );  
%runCF(datafile_cf, u_train, u_test, 'SVDpp' );  
%runCF(datafile_cf, u_train, u_test, 'SLIM' );
%runCF(datafile_cf, u_train, u_test, 'ItemBased' );
%runCF(datafile_cf, u_train, u_test, 'Fabio13' );

%% Multiclass Co-Clustering
    
      fprintf('======================\n');
      fprintf('* Subgroup Num is : %d\n', ClassNum);
      runMCOC_CF(datafile,'M', u_train, u_test, Vdata, ClassNum,dim, 'NMF', VM, UM);
      %runMCOC_CF(datafile,'M', u_train, u_test, Vdata, ClassNum,dim, 'POP');
      %runMCOC_CF(datafile,'M', u_train, u_test, Vdata, ClassNum,dim, 'UserBased');
      %runMCOC_CF(datafile,'M', u_train, u_test, Vdata, ClassNum,dim, 'PMF');
      %runMCOC_CF(datafile,'M', u_train, u_test, Vdata, ClassNum,dim, 'BPMF');
      %runMCOC_CF(datafile,'M', u_train, u_test, Vdata, ClassNum,dim, 'PPMF');
      %runMCOC_CF(datafile,'M', u_train, u_test, Vdata, ClassNum,dim, 'SVD');
      %runMCOC_CF(datafile,'M', u_train, u_test, Vdata, ClassNum,dim, 'SVDpp');
      %runMCOC_CF(datafile,'M', u_train, u_test, Vdata, ClassNum,dim, 'SLIM');
      %runMCOC_CF(datafile,'M', u_train, u_test, Vdata, ClassNum,dim, 'ItemBased');
      %runMCOC_CF(datafile,'M', u_train, u_test, Vdata, ClassNum,dim, 'Fabio13');
      fprintf('======================\n');
end

            
              
