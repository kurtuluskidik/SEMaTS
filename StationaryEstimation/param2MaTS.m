function [A,B,dA,dB] = param2MaTS(theta,M,N,L,p);
% param2MaTS converts the parameter vector into the autoregressive matrices in a
% matrix time series model
%
% SYNTAX:  [A,B,da,dB] = param2MaTS(theta,M,N,L,p)
%
% INPUTS:  theta ... d dimensional real vector of parameters.
%          M     ... integer, row dimension.
%          N     ... integer, column dimension.
%          L     ... integer; number of lags. 
%          p     ... integer, number of terms.
%
% OUTPUTS: A  ... M x M x L x p array of real matrices. 
%          B  ... N x N x L x p array of real matrices.
%          dA ... N x N x L x p x d array of real matrices of derivatives.
%          dB ... N x N x L x p x d array of real matrices of derivatives.
%
% REMARKS: 
% used in MaTS Y_t = \sum_{l=1}^L \sum_{j=1}^p A_{l,j} Y_{t-l} B_{l,j}' + E_t.
% provides parameterized (using the QR decomposition) 
% matrices for p terms.
% Derivatives are provided, if 4 outputs are requested. 
% 
% AUTHOR: dbauer, 21.8.2023.
nth = length(theta);
A = zeros(M,M,L,p);
B = zeros(N,N,L,p);
if nargout>2
    dA = zeros(M,M,L,p,nth);
    dB = zeros(N,N,L,p,nth);
end

npar_term = p*(M*M+N*N-p);

for l=1:L
    index_l = (npar_term)*(l-1)+[1:npar_term];
    theta_l = theta(index_l);
    if nargout>2
        [A(:,:,l,:),B(:,:,l,:),da,db] = param_term(theta_l,M,N,p);
        dA(:,:,l,:,index_l) = da; 
        dB(:,:,l,:,index_l) = db; 
    else 
        [A(:,:,l,:),B(:,:,l,:)] = param_term(theta_l,M,N,p);
    end
end
