%% VECM to VAR
% A_var
% M Companion matrix
function [A_var, M] = VECM_Coef2VAR_Coef(alpha, beta, Gamma)
%
% VECM_Coef2VAR_Coef   Function to converts VECM to VAR
%
% Syntax:
%
%   [A_var, M] = VECM_Coef2VAR_Coef(alpha, beta, Gamma);
%
% ------------------------------------------------------------
% Description:
%
%   VECM_Coef2VAR_Coef       
% -------------------------------------------------------------------------
% Input Arguments:
%
%   alpha - matrix
%
%   beta - matrix
%   
%   Gamma - 3-D Array.
%
% -------------------------------------------------------------------------
% Output Arguments:
%
%    A_var - .
%    M - .
%
% -------------------------------------------------------------------------
% NOTE: 
% -------------------------------------------------------------------------
% Example:
%
%   % Data
%
% -------------------------------------------------------------------------
% References:
% 
%   [1] Kidik, K. and Bauer, D. "Specification and Estimation of Matrix 
%       Time Series Models with Multiple Terms". Preprint.
%
% -------------------------------------------------------------------------
%
% Date:         01.08.2025
% written by:   Kurtulus Kidik 
%               Econometrics
%               Universität Bielefeld
%               Universitätsstraße 25
%               D-33615 Bielefeld
%               kurtulus.kidik@uni-bielefeld.de
% ------------------------------------------------------------------------- 
if nargin<3
    Gamma = [];
end
%[M,N,L,p] = size(Gamma);
dims = size(Gamma);

K1 = size(alpha,1);
K2 = size(beta,1);
if K1 ~= K2
  disp('!!! Attention !!! : Check Dimension !!!, please press any key to continue !!!')
  pause;  % Wait for user to type any key
else
      % continue
end
%L = dims(:,3);
%p = dims(:,4);
%M = size(Gamma,1);
%N = size(B,1);

if length(dims)>=3
    L = (dims(3));
elseif dims(1) == 0
    L = 0;
else
    L = 1;
end
%{
 if length(dims)==4
    p = dims(4);
else
    p = 1;
end
%}
%Gamma(:,:,L) = Gamma;
Gam_temp = zeros(K1,K2,(L+2));
Gam_temp(:,:,1) = -(eye(K1) + alpha*beta');
%Gam_temp(:,:,1) = -(alpha*beta');
if L>0
    Gam_temp(:,:,2:end-1) = Gamma;
else
    Gam_temp(:,:,2:end) = Gamma;
end
%Gam_temp(:,:,end) = zeros(:,:,1);

coefficientMatrix = zeros(K1,K2,(L+1));
for i=1:(L+1)
    coefficientMatrix(:,:,i) = -Gam_temp(:,:,i) + Gam_temp(:,:,i+1);
end
longcoefficientMatrix = reshape(coefficientMatrix, K1, K2*(L+1));
companionMatrix = [ longcoefficientMatrix; eye(K1*L) zeros(K1*L, K2)];
%companionMatrix = [longcoefficientMatrix; eye(size(coefficientMatrix, 3) * (L))];
A_var = coefficientMatrix;
M = companionMatrix;
end