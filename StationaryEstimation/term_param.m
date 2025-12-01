function [theta] = term_param(A,B);
% term_param calculates the parameters corresponding to the p terms in (A,B).
%
% SYNTAX: [theta] = term_param(A,B);
%
% INPUTS: A ... M x M x p real array of p terms.
%         B ... N x N x p real array of p terms.
%        
% OUTPUTS: theta ... d vector of parameters.
%
% REMARKS: used in MaTS Y_t = \sum_{j=1}^p A_j Y_{t-1} B_j' + E_t.
% provides parameters (using the QR decomposition) 
% for p terms matrices.
% 
% AUTHOR: dbauer, 21.8.2023.

[N,~,p]=size(B);
M = size(A,1);

% calculate vectorization. 
KP = zeros(M*N,M*N);
Phi = zeros(N*N,M*M); 
        
for k=1:p
   KronProd = kron(squeeze(B(:,:,k)),squeeze(A(:,:,k)));
   % convert to reordered matrix:
   KP = KP + KronProd;
end
%
% Bu kisim Dietmardan degisik
% transform using Phi.
Phi = reorder_KronProd(KP,N,M,N,M);
%{
% Burasi ni ekledim degistirdim -- start
% Reshaping
rA = reshape(KP, [M,N,M,N]);
pA = permute(rA, [2 4 1 3]);
rpA = reshape(pA, N*N,M*M);
Phi = rpA;
% Burasi ni ekledim degistirdim -- end
%}
% calculate QR decomposition 
[Q,R]= qr(Phi');

% take out the relevant matrices
Q = Q(:,1:p);
R = R(1:p,:);


% adjust sign of diagonal entries
signs = sign(diag(R(1:p,1:p)));
Q(:,1:p)=Q(:,1:p)*diag(signs);
R(1:p,:)=diag(signs)*R(1:p,:);

% obtain parameters
theta = ortho2par_LR(Q);
for jr=1:p
    theta(end+[1:(p-jr+1)])=R(jr,jr:p);
end
theta = [theta(:);reshape(R(:,(p+1):end),(N*N-p)*p,1)];

