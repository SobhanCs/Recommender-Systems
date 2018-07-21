
function dat = CF_UserBased( data )
%FILLMEM Summary of this function goes here
%   Detailed explanation goes here
    [uNum mNum] = size(data);
    SIM_Matrix = cosine_sim(data);
    dat = SIM_Matrix * data;
    %dat = dat/max(dat(:));
end 
function SIM_Matrix = cosine_sim(data)
      [uNum mNum] = size(data);
%       meanVec= sum(data,2)./sum(data>0,2);
%       meanMat = repmat(meanVec,1,mNum);
%       data(data>0) = data(data>0) - meanMat(data>0);
      product_matrix = data*data';
      nm = sqrt(sum(data.^2,2));
      normMatrix = max(1e-12, nm * nm');
      SIM_Matrix = product_matrix ./ normMatrix;
      
end



% function dat = CF_UserBased( data )
% %FILLMEM Summary of this function goes here
% %   Detailed explanation goes here
%     [uNum mNum] = size(data);
%     avg = mean(data(data>0));
%     dat = zeros(uNum,mNum);
%     ceil = sparseMat2CellVec(data);
%     SIM_Matrix = memoryBasedModels(ceil,ceil,1,1,mNum); 
%     
%     Mean_Vector = sum(data,2);
%     for j =1:uNum
%        if(Mean_Vector(j)>0)
%            Mean_Vector(j) =  Mean_Vector(j)/sum(data(j,:)>0);
%        else Mean_Vector(j) = avg;
%        end
%     end   
%     for i=1:uNum
%         if( sum(data(i,:))==0 )
%             continue;
%         end
%         %%active = data(i,:);
%         other = data;
%         other(i,:) = [];
%         %%sim = SimVec(active,other);
%         sim = SIM_Matrix(i,:);
%         sim(i) = [];
%         meanvote = Mean_Vector(i);
%         mpred = predictPreferenceMemBased(sim,other,meanvote);
%         dat(i,:) = mpred;
%     end
% 
% end
