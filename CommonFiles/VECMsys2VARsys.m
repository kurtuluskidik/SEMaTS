function varCoef = VECMsys2VARsys(alpha, beta, Gamma)
%
% VECMsys2VARsys    Function to calcuulates VAR coefficients from VECM
% Generates K-dimensional VECM system with given rank
% Syntax:
%
%   [simY,info_Data] = GenECMAR_data_Mark4(T,M,N,CovSet,p,ra,Re,SP);
% ------------------------------------------------------------
% Description:
%
%   GenVECMdata     Function for generating data for Cointegrated 
%           Vector Autoregresive Time Series for VECM estimation
% ------------------------------------------------------------
% Input Arguments:
%   
%   ra - rank ofPi matrix, alpha, beta 
%   p .......... lag of delta part
%   alpha - 
%   beta - 
%   Gamma - Short run coefficients, Differenced part
%   
% ------------------------------------------------------------
% Output Arguments:
%
%   simY - Data representing observations of a multi-dimensional time series
%       y(t), specified as a numObs-by-numDims matrix.
%
%   checkst - Includes the Trace and Maximum eigenvalue critical values
%   CompMatrix
%   Lambdas
% ------------------------------------------------------------
% Example:
%
%   % Data 
% ------------------------------------------------------------
% Notes:
%
% ------------------------------------------------------------
% References:
% 
%   [1] Johansen, S. Likelihood-Based Inference in Cointegrated Vector
%       Autoregressive Models. Oxford: Oxford University Press, 1995.
%
%   [2] 
%
%   [3] 
% ------------------------------------------------------------
% Date:         01.08.2025
% written by:   Kurtulus Kidik 
%               Dept. of Economics
%               Universität Bielefeld
%               Universitätsstraße 25
%               D-33615 Bielefeld
%               kurtulus.kidik@uni-bielefeld.de
% ------------------------------------------------------------
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

if length(dims)>=3
    L = (dims(3));
elseif dims(1) == 0
    L = 0;
else
    L = 1;
end
p = L+1;
Gamma(:,:,p) = zeros(K1,K1);
varCoef = zeros(K1,K2,p);
if L == 0
varCoef(:,:,1) = eye(K1) + alpha*beta';
else
    varCoef(:,:,1) = eye(K1) + alpha*beta' + Gamma(:,:,1);
    for ii = 1 : L
        varCoef(:,:,ii+1) = - Gamma(:,:,ii) + Gamma(:,:,ii+1);
    end
end
end % END VECMsys2VARsys