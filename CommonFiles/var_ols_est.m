%% VAR OLS estimation witout deterministic variables 
function [VAROlsEst_Results,VAROlsEst_info] = var_ols_est(y,p,plots)
%
% function [coeffs_array,coeffs,res,aic,bic] = var(y,p)
%
% Function to estimate VAR with p
% 
% Normalization: a_0 = 0,
% y(t) = a1*y(t-1) + a2*y(t-2) +... + ap*y(t-p) + e_t
%
% Input:               y ... Observations, T by s
%                      p ... Order of the VAR
%                      plots ... Plots residuals.i.e.: 1
%
% Output:   coeffs_array ... s by s by p array of coefficients
%                    Phi ... [a1,...,ap]
%                 resids ... Residuals, T by s
%                    aic ... Akaike Information Crit. 
%                    bic ... Akaike Information Crit.
% 
% NOTE: No deterministic variables included in this version!
%-----------------------------------------------------------
% AUTHOR: Kurtulus Kidik, 11.06.2024. 
%-----------------------------------------------------------
if nargin<3
   plots = 0;
end
[T,s] = size(y);
Teff = T-p;
%if p<0 % if negative lag length.
%   p = abs(p);
%end
VAROlsEst_info = struct('CodeName','var_ols_est.m','DataSizeT',T,'Dimensions',s,'NumberOfLags',p,'Data',y);
%Generate lagged regressor matrix
z = zeros(T,s*p);

for i = 1:p
    z(:,(1+(i-1)*s):i*s) = [NaN(i,s );y(1:end-i,:)];    
end


%---
y_eff = y(p+1:end,:);
z_eff = z(p+1:end,:);

% Regression
coeffs = (z_eff \ y_eff);
res = y_eff - z_eff*coeffs;

%Order estimation AIC and BIC
VCV = res'*res;
Omega = VCV/(Teff);
logdetOmega = log(det(Omega));
NOPar = p*s*s;

%AIC: Akaike Information Criterion 
ic_aic = logdetOmega + (2*NOPar)/Teff;

%AIC: Corrected Akaike Information Criterion 
ic_Caic = ic_aic + (2*p)*(p+1)/(T-p-1); 

%BIC: Schwarz - Bayesian Information Criterion  
ic_bic = logdetOmega + (NOPar*log(Teff))/Teff;

%HQC : Hannan-Quinn Criterion
ic_hqc = logdetOmega + NOPar*(2*log(log(Teff))/Teff);

%Coefficients as estimated above are "transposed", thus:
Phi = coeffs';

%Write down coefficients as an array if wanted
coeffs_array = reshape(Phi,s,s,p);

% log likelihood
LL =  -0.5*((Teff)*logdetOmega+(Teff)*s);
%LL =  (T-p-1)*log(det(res'*res/(T-p-1)))+(T-p-1)*s;

resids = [NaN(p,s);res];

[~,ComMatrix] = VARCompMatrix(coeffs_array);
EigenValCM = abs(eig(ComMatrix));
LambdaCM = max(EigenValCM);
StabilityCheck = lt(LambdaCM,1);
VAROlsEst_Results = struct('CodeName','var_ols_estimate.m','LoglikelihoodValue',LL,...
    'SigmaHat',Omega,'LogDetSigmaHat',logdetOmega,'VARCoefficients',coeffs_array,'Residuals',res,...
    'VarAIC',ic_aic,'VarAICc',ic_Caic,'VarSBIC',ic_bic,'VarHQC',ic_hqc,'SampleSizeT',T,...
    'Dimensions',s,'NumberOfLags',p,'Df',NOPar,'Stability',StabilityCheck);
if plots
    figure;
    hist(res,20);
    title('Residuals VAR(p)')

end
end


