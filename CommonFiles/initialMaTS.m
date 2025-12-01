%% Initial Esimation 
function [A,B,SigmaR,SigmaC] = initialMaTS(Y,p,J)
% Function to estimates the initial values for MaTS with VAR
%
% initialMaTS provides initial values for coefficient matrices for 
%               MaTS estimations
% Syntax:
%
%   [A,B] = initialMaTS(Y,p,J);
%   [A,B,SigmaR,SigmaC] = initialMaTS(Y,p,J);
%
% -------------------------------------------------------------------------
% Description:
%
%   MaTS estimations needs good initial values. One way, VAR estimators 
%       can be used. Function uses nearest Kronecker product approximation 
%       with QR decomposition.  
%           
% -------------------------------------------------------------------------
% Input Arguments:
%
%   Y - Data representing observations of a multi-dimensional time series
%       Y(t), specified as a three dimensional array of dimension T x M x N. 
%       T: Time dimension, R: Row dimension, M: Column dimension.
%       The last observation is the most recent.
%   p - integer; lag length.
%   J - integer; number of terms per lag. i.e. 2
%
% -------------------------------------------------------------------------
% Output Arguments:
%
%   A - Row coefficent matrix, four dimensional array M x M x p x J.
%   A - Column coefficent matrix, four dimensional array M x M x p x J.
%   SigmaR - Row covariance matrix, M x M.
%   SigmaC - Column covariance matrix, N x N.
%
% -------------------------------------------------------------------------
% References:
% 
%   [1] Kidik, K. and Bauer, D. "Specification and Estimation of Matrix 
%       Time Series Models with Multiple Terms". Preprint.
%
% -------------------------------------------------------------------------
%
% AUTHOR:
% Kurtulus KIDIK, 
% Econometrics
% Universität Bielefeld
% Universitätsstraße 25
% D-33615 Bielefeld
% kurtulus.kidik@uni-bielefeld.de
% 21.06.2024.
% -------------------------------------------------------------------------

if nargin<3
    J = 1;    
end
if nargin<2
    p = 1;    
end
[T,M,N] = size(Y);
%
if N <= 1
  error('!!! Attention !!! :Y is Not three dimensional array data !!! Please check dimensions')
end
%
K = M*N;
y = reshape(Y,T,K);
%
% Estimate VAR model
[VAROlsEst_Results,~] = var_ols_est(y,p);
%
C = VAROlsEst_Results.VARCoefficients;
[Ms,Ns,p] = size(C);
%
% Make sure dimensions match.
if (Ms ~= M*N)
    error("Check dimensions for A!")
end
if (Ns ~= N*M)
    error("Check dimensions for B!")
end
A = zeros(M,M,p,J);
B = zeros(N,N,p,J);
for ii=1:p
%{
    Phi = reorder_KronProd(squeeze(C(:,:,j)),N,M,N,M);
    [Q,R]= qr(Phi');

    % adjust sign of diagonal entries
    signs = sign(diag(R(1:p,1:p)));
    Q(:,1:p) = Q(:,1:p)*diag(signs);
    R(1:p,:) = diag(signs)*R(1:p,:);
    % write result into matrices
    for k=1:p
        A(:,:,j,k) = reshape(Q(:,k),M,M);
        B(:,:,j,k) = reshape(R(k,:),N,N)';
    end
%}
%
    [B1, A1] = NKroProQR(squeeze(C(:,:,ii)), [N N], [M M], J);
    for ik=1:J      
        A(:,:,ii,ik) = squeeze(A1(:,:,ik));
        B(:,:,ii,ik) = squeeze(B1(:,:,ik));
    end
%
end
%
% If needed; Covariance Matrix is Kronecker product. NKP approximation of SigmaR and SigmaC 
Omega = VAROlsEst_Results.SigmaHat;
[SigmaC, SigmaR] = NKroProQR(Omega, [N N], [M M], 1);
%{
Reordered_Omega = reorder_KronProd(Omega,N,M,N,M);
[Qs,Rs]= qr(Reordered_Omega');
signsigma = sign(diag(R(1,1)));
Qsigma = Qs(:,1)*diag(signsigma);
Rsigma = diag(signsigma)*Rs(1,:);
SigmaR = reshape(Qsigma,M,M);
SigmaC = reshape(Rsigma,N,N)';
%}
end
