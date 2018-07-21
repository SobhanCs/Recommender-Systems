function [ Precision,Recall,F1,MAP,NDCG ] = evaluation_macro( prediciton, test_table )
%EVALUATION_MACRO Summary of this function goes here
%   Detailed explanation goes here
topN = 200;     % top 200 recommended items are evaluated
[N,M] = size(prediciton);
pres_table = zeros(N,topN);%% save hit count for each user
recall_table = zeros(N,topN);
f1_table = zeros(N,topN);
corr_table = zeros(N,topN);
ndcg_table = zeros(N,topN);
AP = zeros(N,1);

[sortR,sortIdx] = sort(prediciton,2,'descend'); % Ranking the scores for each user
for user = 1:N
    %user
    if(sum(test_table(user,:))>0)
        testItems = test_table(user,:);
        relevantNum = sum(testItems>0);
        for r = 1:topN 
           if(  testItems( sortIdx(user,r)) >0 )
                corr_table(user,r) = 1;
           else corr_table(user,r) = 0;
           end
        end
        for rec = topN:-1:1
            recNum = rec ;
            hitCount = sum(corr_table(user,1:rec)>0);
            if(hitCount == 0)
                pres_table(user,rec)=0;recall_table(user,rec)=0;f1_table(user,rec)=0;
                continue;
            end
            p =hitCount/recNum;
            pres_table(user,rec) = p;
            r = hitCount /relevantNum;
            recall_table(user,rec) = r;
            f1_table(user,rec) = 2*p*r/(p+r);
        end
        if(sum(corr_table(user,:))>0)
            AP(user) = pres_table(user,:) * corr_table(user,:)' / sum(corr_table(user,:));
        else AP(user) = 0;
        end
        for atK =1:topN
            IDCG = sum( (2.^(ones(1,atK)) -1) ./ (log2((1:atK) + 1))  )  ;
            DCG = sum( (2.^(corr_table(user,1:atK)) -1) ./ (log2((1:atK) + 1))  ) ;
            ndcg_table(user,atK) = DCG/IDCG;
        end
    end
end
Precision = mean(pres_table)';
Recall = mean(recall_table)';
F1 = mean(f1_table)';
NDCG = mean(ndcg_table)';
MAP =  mean(AP);
end

