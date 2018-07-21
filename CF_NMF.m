function [ data ] = CF_NMF( u_train, VM, UM, U, c, u_test )
    
    [N,M] = size(u_test);
    data = VM(U(c,1:N)>0,:)*UM(:,U(c,(N+1):end)>0);

return