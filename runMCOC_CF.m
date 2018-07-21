function runMCOC_CF(datafile,type, u_train, u_test,Vdata, ClassNum,dim, method, VM, UM )    
[N,M] = size(u_test);
groups = ceil(log2(ClassNum))+1; % # of subclusters for each item
    
        % STEP2 : multiclass clustering
              %%www-2012 vertion
                default_options = [2;	% exponent for the partition matrix U
                100;	% max. number of iteration
                1e-6;	% min. amount of improvement
                0];	% info display during iteration 
                %tic;
                %[Centers, U, Obj] = fcm(Vdata, ClassNum,default_options) ; 
                %toc;

U = Vdata';
                
                for i =1:N+M
                        [ud, uidx] = sort(U(:,i),'descend');
                        U(uidx((groups+1):end),i) = 0;
                        U(:,i) = U(:,i)/sum(U(:,i));
                end
                
    
    Biclusts = cell(1,size(U,1));
        for c=1:size(U,1)
            biclust = u_train(U(c,1:N)>0,U(c,(N+1):end)>0);
            if(size(biclust,1)<2 || size(biclust,2)<2 )
                biclust = ones(size(biclust));
            else
                try
                    %learnname = sprintf('CF_%s(biclust)', method);
                    %[biclust] = eval(learnname); % Any CF method to fill the sparse biclust to be a full one
                    %[biclust] = CF_NMF(biclust, VM, UM, U, c, u_test);
                    [hr,arhr,biclust] = recom_mc(biclust,u_test,u_test,1e-3,5)
                catch
                end
            end
            Biclusts{c} = biclust;% .* ClassNum;
        end
    
index = 5;
pred = zeros(size(u_test));
%predic = zeros(size(u_test));
pred = merge(N, U, Biclusts, pred, index, ClassNum);  %5 - merge

pred(u_train>0) = 0;

% Recommendation Evaluation
    [Precision,Recall,F1,MAP,NDCG] = evaluation_macro(pred, u_test);  %6 - evaluation_macro
    fprintf('MCOC::method = %s,  ', method);
    fprintf('P@10 = %f\n', Precision(10));
    
    fold = strcat('dim',num2str(dim));
    if( ~exist(strcat('nmf_results\',fold), 'dir') )
        mkdir('nmf_results',fold);
    end
    if( ~exist(strcat('nmf_results\',fold,'\',num2str(ClassNum)), 'dir') )
        mkdir(strcat('nmf_results\',fold),num2str(ClassNum));
    end
    savename = strcat('nmf_results\',fold,'\',num2str(ClassNum),'\',datafile,'_MCOC_',method,'.mat');
    save(savename, 'Precision', 'Recall', 'F1', 'MAP', 'NDCG', 'U',...
                  'ClassNum');
    
    A(1)= Precision(10);
    A(2)= NDCG(10);
    A(3)= F1(10);
    A(4)= MAP;
    
    %sheet = sprintf('%s', method);
    %xlRange = sprintf('B%.0f',ClassNum);
    %xlRange = 'B2';
    %xlswrite('mex.xls',A,sheet,xlRange)
    xlswrite(sprintf('result%.0f',ClassNum),A)
end