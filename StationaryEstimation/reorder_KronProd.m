function Phi = reorder_KronProd(KronProd,M,N,Mz,Nz)
% reorderKronProd reorders the entries in the matrix such that a Kronecker
% product kron(A,B) for two square matrices B (NxNz) and A (MxMz) equals vA vB'.
%
% SYNTAX: Phi = reorder_KronProd(KronProd,M,N,Mz,Nz)
%
% INPUTS:    KronProd  ... MN x MzNz matrix. 
%            M         ... integer;
%            N         ... integer; 
%
% OUTPUT:    Phi  ... M Mz x N Nz matrix. 
%
% AUTHOR: dbauer, 6.6.2023.

if nargin<5
    Nz = N;
    Mz = M;
end

Phi = zeros(M*Mz,N*Nz); 

dims = size(KronProd);
if ((dims(1) == M*N)+(dims(2) == Mz*Nz)) ~= 2
    error("Matrix not of correct size!")
end

for a=1:M
    for b=1:Mz
        mat = KronProd((a-1)*N+(1:N),(b-1)*Nz+(1:Nz));
        Phi((a-1)*Mz+b,:) = reshape(mat,1,N*Nz);
    end
end

end