function [A,B,dA,dB] = param_term(theta,M,N,p)
% param_term converts the parameter vector into the sum of r terms in a
% matrix time series model
%
% SYNTAX:  [A,B,da,dB] = param_term(theta,M,N,p)
%
% INPUTS:  theta ... d dimensional real vector of parameters.
%          M     ... integer, row dimension.
%          N     ... integer, column dimension.
%          p     ... integer, number of terms.
%
% OUTPUTS: A  ... M x M x p array of real matrices. 
%          B  ... N x N x p array of real matrices.
%          dA ... N x N x p x d array of real matrices of derivatives.
%          dB ... N x N x p x d array of real matrices of derivatives.
%
% REMARKS: 
% used in MaTS Y_t = \sum_{j=1}^p A_j Y_{t-1} B_j' + E_t.
% provides parameterized (using the QR decomposition) 
% matrices for p terms.
% Derivatives are provided, if 4 outputs are requested. 
% 
% AUTHOR: dbauer, 21.8.2023.

nth = length(theta);
M2 = M^2; 
ntriang = p*(p-1)/2;
npar = p*(M2-p)+ntriang;
par = theta(1:npar); 
theta(1:npar)=[];
% calculate Q.
[Q] = par2ortho_LR(par,M^2,p);
for jr = 1:p
    A(:,:,jr) = reshape(Q(:,jr),M,M);
end

% calcuate R
R = zeros(p,N*N);
for j=1:p
    R(j,j:p) = theta(1:(p-j+1));
    theta(1:(p-j+1)) = [];    
end

R(:,(p+1):end) = reshape(theta(1:((N*N-p)*p)),p,(N*N-p));
for jr = 1:p
    B(:,:,jr) = reshape(R(jr,:)',N,N)'; % burasi dietmardan 
    %B(:,:,jr) = reshape(R(jr,:)',N,N); % bununla degistirdim.

end

% calculate derivatives 
if nargout>2
    dA = zeros(M,M,p,nth);
    dB = zeros(N,N,p,nth);
    for j=1:npar
        dQ = dpar2ortho_LR(par,M^2,p,j);
        for jr = 1:p
            dA(:,:,jr,j) = reshape(dQ(:,jr),M,M);
        end
    end
    cur = npar+1;
    dRo = zeros(p,N*N);
    for jr = 1:p
        for jc=jr:p
            dR = dRo;
            dR(jr,jc)=1;
            dB(:,:,jr,cur) = reshape(dR(jr,:)',N,N)';
            cur = cur+1;
        end
    end
    for jc=(p+1):(N*N)
        for jr = 1:p
            dR=dRo;
            dR(jr,jc)=1;
            dB(:,:,jr,cur) = reshape(dR(jr,:)',N,N)';
            cur = cur+1;
        end
    end
end

