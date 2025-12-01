%% Function to calculate companion matrix for VAR system
function [A_var, companionMatrix] = VARCompMatrix(VarCoef);
%
% VARCompMatrix    Computes compation matrix of VAR system
%
% Syntax:
%
%   [A_var, companionMatrix] = VARCompMatrix(VarCoef);
% ------------------------------------------------------------
% Description:
%
%   VARCompMatrix    Computes compation matrix of VAR system
% ------------------------------------------------------------
% Input Arguments:
%
%   VarCoef - Three dimensional Coefficient array. K x K x L.
%   
% ------------------------------------------------------------
% Output Arguments:
%
%   A_var - Returns Coefficient arrray 
%   companionMatrix - Computed Companion Matrix.
%   
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
%   [2] Kilian L, Lütkepohl H. Structural Vector Autoregressive Analysis. 
%       Cambridge University Press; 2017.
% 
% ------------------------------------------------------------
% Date:         01.08.2025
% written by:   Kurtulus Kidik 
%               Dept. of Economics
%               Universität Bielefeld
%               Universitätsstraße 25
%               D-33615 Bielefeld
%               kurtulus.kidik@uni-bielefeld.de
% ------------------------------------------------------------
%

    dims = size(VarCoef);   
    K1 = size(VarCoef,1);
    K2 = size(VarCoef,2);
    
    % Determine the number of lags
    if length(dims) >= 3
        L = (dims(3));
    else
        L = 1;
    end
    
    longcoefficientMatrix = reshape(VarCoef, K1, K2*L);
    companionMatrix = [ longcoefficientMatrix; eye(K1*(L-1)) zeros(K1*(L-1), K2)];
    A_var = VarCoef;

end