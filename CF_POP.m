function dat = CF_POP( data )
%FILLMEM Summary of this function goes here
%   Detailed explanation goes here
    [N,M] = size(data);
    dat = ones(N,1) * sum(data>0);
end