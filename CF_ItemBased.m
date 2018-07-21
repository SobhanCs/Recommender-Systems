function [prediction ] = CF_ItemBased( u_train )
%ITEMBASED_CF Summary of this function goes here
%   Detailed explanation goes here
    SIM_Matrix = SimItem(full(u_train));
    prediction = u_train * SIM_Matrix;
    
    
    function SIM_Matrix = SimItem(data)  %cosine        
      [uNum mNum] = size(data); 
      SIM_Matrix = eye(mNum,mNum);
      product_Matrix = data'*data  ;
      nm= sqrt(sum(data.*data));
      Nm_matrix = nm'*nm;
      for i = 1:mNum
          %i
          for j=(i+1):mNum
              if(product_Matrix(i,j) ~=0)
                 SIM_Matrix(i,j) = product_Matrix(i,j)/(Nm_matrix(i,j));
                 SIM_Matrix(j,i) = SIM_Matrix(i,j);
              end
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