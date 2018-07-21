function [ data] = CF_SVDpp( u_train, imp_data )
    tmp_T = tic;
    if (~exist('imp_data','var'))
       imp_data = zeros(size(u_train));
    end    
    r = 6;
    [N,M] = size(u_train);
    rm = zeros(N,1);
    mm = mean(u_train(u_train>0));    
    for i=1:N
        if(sum(u_train(i,:))>0)
            rm(i) = mean(u_train(i,u_train(i,:)>0))-mm; 
        else 
            rm(i) = 0;
        end
    end
    cm = zeros(1,M);
    for i=1:M
        if(sum(u_train(:,i))>0)
            cm(i) = mean(u_train(u_train(:,i)>0,i))-mm;
        else
            cm(i) = 0;
        end
    end
    data = u_train - ones(N,1)*cm;
    data = data - rm*ones(1,M);
    
%     data = u_train + ones(N,1)*cm;
%     data = data - rm*ones(1,M);
%     data(u_train == 0) = 0;    

    imp_data = double(u_train(:,:)>0);    
    
    [U,S,V] = svds(data, r);
    S2= S.^(1/2);
    U = U*S2;
    V = V*S2;
    
    VV = pinv(U)*imp_data;
    
    for i=1:N
        tmp = zeros(1, r);
        tmp2 = 0;
        for j=1:M
            if (imp_data(i,j)>0)
                tmp = tmp + VV(j);
                tmp2 = tmp2 + 1;
            end
        end
        if (tmp2>0)
            U(i,:) = U(i,:)+tmp*(tmp2^(-2));
        end
    end
    data = U*V'+rm*ones(1,M)+ones(N,1)*cm;
    
    elapse = toc(tmp_T);
return