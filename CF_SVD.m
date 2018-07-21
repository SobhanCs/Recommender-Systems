function [ data] = CF_SVD( u_train, r )
tmp_T = tic;
    if (~exist('r','var'))
       r = 6;
    end
    
    [N,M] = size(u_train);
    rm = zeros(N,1);
    for i=1:N
        if(sum(u_train(i,:))>0)
            rm(i) = mean(u_train(i,u_train(i,:)>0)); 
        else 
            rm(i) = 4;
        end
    end
    cm = zeros(1,M);
    for i=1:M
        if(sum(u_train(:,i))>0)
            cm(i) = mean(u_train(u_train(:,i)>0,i));
        else
            cm(i) = 4;
        end
    end
    data = u_train + ones(N,1)*cm;
    data = data - rm*ones(1,M);
    
%     data = u_train + ones(N,1)*cm;
%     data = data - rm*ones(1,M);
%     data(u_train == 0) = 0;

    [U,S,V] = svds(data,r);
    S2= S.^(1/2);
    U = U*S2;
    V = V*S2;
    data = U*V'+ rm*ones(1,M);
elapse = toc(tmp_T);
return