
function [prediction ] = CF_REC13( u_train )


    a = 0.85;
    b = 0.7;
    c = 0.5;
    q = 4;

    [uNum mNum] = size(u_train);
%     for i = 1:uNum
%         for j = 1:mNum
%             if (u_train(i,j)>0)
%                 u_train(i,j) = 1;
%             end
%         end
%     end

    u_train = double(u_train(:,:)>0);
   
    
    
    data = u_train;
    [uNum mNum] = size(data); 
    SIM_Matrix = eye(mNum,mNum);
    product_Matrix = data'*data  ;
    nm= sum(data.*data);    
    Nm_matrix = (nm.^a)'*(nm.^(1-a));
    for i = 1:mNum
        %i
        for j=(i+1):mNum
            if(product_Matrix(i,j) ~=0)
               SIM_Matrix(i,j) = product_Matrix(i,j)/(Nm_matrix(i,j));
               SIM_Matrix(j,i) = product_Matrix(j,i)/(Nm_matrix(j,i));
            end
        end
    end
    
    SIM_Matrix = SIM_Matrix.^q;
    
    nm_u = sum(data, 2);
    sw = sum(SIM_Matrix.^2, 2);
     
    prediction = u_train * SIM_Matrix;
    Nm_matrix2 = (nm_u.^b)*((sw').^(1-b));
    
    prediction = prediction./Nm_matrix2;
%     for i = 1:uNum
%         %i
%         for j=1:mNum
%             if(prediction(i,j) ~=0)
%                prediction(i,j) = prediction(i,j)/(Nm_matrix2(i,j));
% %                prediction(j,i) = prediction(j,i)/(Nm_matrix2(j,i));
%             end
%         end
%     end

    
    u_train2 = double(~u_train);
    prediction2 = prediction.*u_train2;
    
%     prediction2 = prediction;
%     
    nm_max = max(prediction2')';
    nm_avg = sum(prediction2, 2);
    nm_num = zeros(uNum, 1);
    for i = 1:uNum
        for j = 1:mNum
            if (prediction2(i,j)>0)
                nm_num(i) = nm_num(i)+1;
            end
        end
    end
    nm_avg = nm_avg./nm_num;
    
    for i = 1:uNum
        for j = 1:mNum
            
%             if (prediction(i,j)>nm_max(j))
%                 fprintf('...');
%             end
                
            
            if (prediction(i,j)<nm_avg(i))
                prediction(i,j) = prediction(i,j)/nm_avg(i)*c;
            elseif (prediction(i,j)<nm_max(i))
                prediction(i,j) = c+ (prediction(i,j)-nm_avg(i))/(nm_max(i)-nm_avg(i))*(1-c);
                else prediction(i,j) = 1;
            end
        end
    end
           

   
end




% function [P ] = CF_ItemBased( data )
% %ITEMBASED_CF Summary of this function goes here
% %   Detailed explanation goes here
%     [uNum,mNum] = size(data); 
%     S = corr(data);
%     
%     P = data * S;
%     for u=1:uNum
%         sumOfWeight = max(1e-12, sum(abs(S(data(u,:)>0,:))));
%         P(u,:) = P(u,:)./sumOfWeight;
%     end
%   
% end
% function [similarity] = corr(R)
%     [u,m] = size(R);
%     meanOfItem = sum(R) ./ sum(R>0);
%     M = repmat(meanOfItem,u,1);
%     nR = R;
%     nR(R>0) = R(R>0) - M(R>0);
%     similarity = cosine_sim(nR');
% end 
% 
% function SIM_Matrix = cosine_sim(data)  %row vectors
%       [uNum mNum] = size(data);
%       product_matrix = data*data';
%       nm = sqrt(sum(data.^2,2));
%       normMatrix = max(1e-12, nm * nm');
%       SIM_Matrix = product_matrix ./ normMatrix;
% end