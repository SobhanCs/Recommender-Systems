function A = merge(N, U, Biclusts, pred, index, ClassNum)
%  STEP4 : Merge all biclusters' predicitons into the matrix 'pred'

switch(index)
    case 1
        for i=1:N
            Col_i = U(:,i);
            Col_i(Col_i ==0) = 1e6;
            while(sum(Col_i<1e6)>0)
                [wt,c] = min(Col_i);
                tmp = U(c,(N+1):end)>0;
                biclust1 = Biclusts{c};
                pred(i,tmp) = biclust1( sum(U(c,1:i)>0), :);
                Col_i(c) = 1e6;
            end
        end
        
    case 2
        for i=1:N
            Col_i = U(:,i);
            Col_i(Col_i ==0) = 1e6;
            while(sum(Col_i<1e6)>0)
                [wt,c] = min(Col_i);
                tmp = U(c,(N+1):end)>0;
                biclust1 = Biclusts{c};
                pred(i,tmp) = pred(i,tmp)+wt* biclust1( sum(U(c,1:i)>0), :);
                Col_i(c) = 1e6;
            end
        end 
        
    case 3
        for i=1:N
            Col_i = U(:,i);
            Col_i(Col_i ==0) = 1e6;
            while(sum(Col_i<1e6)>0)
                [wt,c] = min(Col_i);
                tmp = U(c,(N+1):end)>0;
                TT=wt+U(c,tmp);
                TT=TT;
                biclust1 = Biclusts{c};
                pred(i,tmp) = pred(i,tmp)+TT.* biclust1( sum(U(c,1:i)>0), :);
                Col_i(c) = 1e6;
            end
        end
        
    case 4
        for i=1:N
            Col_i = U(:,i);
            for j=1:size(U,2)-N
                [wt, c] = max ( Col_i .* U(:,(N+j)) );
                if wt~=0
                    biclust1 = Biclusts{c};
                    l=find(U(c,:)>0);
                    l1 = find(l==i);
                    l2 = find(l==j+N);
                    l2=l2-size(biclust1,1);
                    pred(i,j) = biclust1(l1, l2);
                end
            end
        end
        
    case 5
        for i=1:N
            for j=1:size(U,2)-N
                tmpp = U(:,i).* U(:,(N+j));
                if(sum(tmpp)~=0)
                    tmpp=tmpp/sum(tmpp);
                end
                for c=1:ClassNum   %the number of classes
                    if (tmpp(c)~=0)
                        biclust1 = Biclusts{c};
                        l=find(U(c,:)>0);
                        l1 = find(l==i);
                        l2 = find(l==j+N);
                        l2=l2-size(biclust1,1);
                        pred(i,j) = pred(i,j) + biclust1(l1, l2);
                        %predic(i,j) = predic(i,j) + pred(i,j);
                    end
                end
            end
        end
        
    otherwise
        disp('other value')
end
A = pred;
