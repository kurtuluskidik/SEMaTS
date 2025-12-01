function [c,ceq] = StabCon(x,M,N,L,p)
%UNTITLED3 Summary of this function goes here
%   Detailed explanation goes here
[Aest,Best] = param2MaTS(x,M,N,L,p);

Phi = zeros(M*N,M*N,L); 
for l=1:L
    for jr=1:p
        Phi(:,:,l) = Phi(:,:,l) + kron(Best(:,:,l,jr),Aest(:,:,l,jr));
    end
end
longcoefficientMatrix = reshape(Phi, (M*N), (M*N)*L);
companionMatrix = [longcoefficientMatrix; eye(M*N*(L-1)) zeros(M*N*(L-1), (M*N))];
Lamda = abs(eig(companionMatrix));
maxEig = max(Lamda);
%c = maxEig-1.000000000000001;
c = maxEig-1;
ceq = [];
end