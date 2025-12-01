%% Quasi-Maximum Likelihood estimation with Statbility costraints
function [MaTSEst_Results,Est_Resuts_Table,Est_fnimcon_output] = ConstQMLE_MaTS(Y,p,j,Ainit,Binit,plots,printresult)
%
% ConstQMLE_MaTS Function to estimate multi-term MAR(p), MARj(p) 
%
% Syntax: 
% 
%   [MaTSEst_Results] = ConstQMLE_MaTS(Y, 2, 3);
%
% -------------------------------------------------------------------------
% Description:
%
%   ConstQMLE_MaTS estimates multi-term MARj(p) model by quasi-maximum 
%           likelihood. Minimizes the variance by the model:
%
%           Y(t) = A_{1,1}*Y(t-1)*B_{1,1}'+ A_{1,2}*Y(t-1)*B_{1,2}'+ ... 
%
%                  + A_{2,1}*Y_(t-2)*B_{2,1}'+ A_{2,2}*Y(t-2)*B_{2,2}' + ... 
%                                               
%                     + A_{p,j}*Y_(t-p)*B_{p,j}' + E_t
%
%   The function uses the same term length for each lag.
% -------------------------------------------------------------------------
% Input Arguments:
%
%   Y1 - Data representing observations of a multi-dimensional time series
%       Y(t), specified as a three dimensional array of dimension T x M x N. 
%       T: Time dimension, R: Row dimension, M: Column dimension.
%       The last observation is the most recent. 
%
%   p - number of lags.
%
%   j - number of terms per lag, here requires. 
%   
%   Ainit - initial values of coefficients. (if not use inital estimation)
%   
%   Binit - initial values of coefficients.
%
%   plots - integer; 1 plots residuals, 2 plots iteration progress.
%
%   printresult - integer; indicating, what information should be
%                          provided in the iterations,0,1,2.
%
% -------------------------------------------------------------------------
% Output Arguments:
%
%   MaTSEst_Results - Results for estimated MARj(p) model. It is a 
%           structure with the following fields that correspond directly 
%           to those in the estimated model:
%
%       EstInfo - Structures of estimation information.
%       CodeName - Name of the MATLAB program.
%       LoglikelihoodValue - Loglikelihood Value.
%       SigmaHat - Covariance Matrix, M*N by M*N. 
%       LogDetSigmaHat - 
%       Ahat - Four dimensional array of Row Coefficients,M x M x p x J 
%       Bhat - Four dimensional array of Column Coefficients,N x N x p x J.
%       VARCoefficients - Three dimensional array of Coefficients,K x K x p.
%       Residuals - (T-p) by K matrix.
%       MatsAIC - Akaike Information Criterion.
%       MatsAICc - Corrected Akaike Information Criterion.
%       MatsSBIC - Schwarz - Bayesian Information Criterion. 
%       MatsAIC - Hannan-Quinn Criterion.
%       MatsAIC - Akaike Information Criterian.
%       SampleSizeT - Number of observations.
%       RowDimension - Number of rows.
%       ColumnDimension - Number of columns.
%       NumberOfLags - 
%       NumberOfTerms - 1 by J vector. i.e. : [2,1].
%       DF - Number of parameters
%       Stability - 
%
%   ExoEst_Results_Table - Table of output. prints Not defined yet.
%
%   ExoEst_details_output - Details of output. prints Not defined yet.
%
% -------------------------------------------------------------------------
% NOTE: No deterministic variables included in this version!
% -------------------------------------------------------------------------
% Example:
%
%   % Data
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
%
if nargin<7
    printresult = 0;
end
if nargin<6
    plots = 0;
end
if nargin<4 
  %  [Ainit,Binit] = est_initial_MaTSbydi(Y,p,j);
  [Ainit,Binit] = est_initial_MaTS(Y,p,j);
end
if nargin<2
    p = 1;
end
if nargin<3
    j = 1;
end

% get dimensions 
[T,M,N] = size(Y);
K = M*N;
Teff = T-p;
ranugeMultiMaTSest = rng;
%Est_info = {ranugeMultiMaTSest, sprintf('Estimation Method: Quasi-Maximumm Likelihood Estimation by ConstQMLE_MaTS.m'),...
%    sprintf('with Sample Size, T: %d,\n Row Dimension: %d,\n, Column Dimension: %d,\n # of Lags: %d,\n # of Terms: %d,\n',T,M,N,p,j),Y};
MaTSEst_info = struct('EstimationMethod','Quasi-Maximumm Likelihood Estimation',...
    'CodeName','ConstQMLE_MaTS.m','SampleSizeT',T,...
    'RowDimension',M,'ColumnDimension',N,'NumberOfLags',p,'NumberOfTerms',j,...
    'RandomNuGen',ranugeMultiMaTSest);
MaTSEst_info.InitialA = Ainit;
MaTSEst_info.InitialB = Binit;
MaTSEst_info.EstimatedData = Y;
MaTSEst_info.readme = 'Output: Est_Resuts,Est_Resuts_Table,Est_fnimcon_output.';
%----------------------------
% INITIALIZION
%----------------------------
% Generate initial values or get initial estimate
%Ainit(:,:,1:L,1:p) = zeros(M,M,L,p);
%Binit(:,:,1:L,1:p) = zeros(N,N,L,p);
%Ainit(:,:,1:L,1:p) = randn(M,M,L,p);
%Binit(:,:,1:L,1:p) = randn(N,N,L,p);
%
%Initial Estimation if needed
%
%[Ainit,Binit] = initialMaTS(Y,p,j);  %*******USE this for initial estimate
%
%Ainit = Aestq;
%Binit = Bestq;
% calucate corresponding parameter vector
x0 = MaTS2param(Ainit,Binit);
x0(abs(x0)<0.0001) = 0.0001;
%x0(x0==0) = 1.0e-7;

% non-linear optimization
options = optimoptions('fmincon','Display','off',...
    'MaxIterations',2500);
options.MaxFunctionEvaluations = 1e+10;
% options.UseParallel = true; % use this if supports
% options.Display = on;
% options.ConstraintTolerance = 1.0e-10;
% options.StepTolerance = 1.0e-6;
% options.OptimalityTolerance = 1.0e-6;
% options.MaxIterations = 100;
% options.Algorithm = sqp;  %'interior-point',"EnableFeasibilityMode",true
% options.EnableFeasibilityMode = true;
% active-set ,'interior-point',"EnableFeasibilityMode",true 
% 9.000000e-01

%[thetaest,fval,exitflag,output,lambda,grad,hessian] = 
[thetaest,fval,exitflag,output,lambda,grad,hessian] = fmincon(@(x)LL_ConQMLE(Y,x,p,j),...
    x0,[],[],[],[],[],[],@(x)StabCon(x,M,N,p,j),options);

% prepare output 
Est_fnimcon_output = struct('theta',thetaest,'qMLEfval',fval,'qMLEexitflag',...
    exitflag,'qMLEoutput',output,'qMLElambda',lambda,'qMLEgrad',grad,'qMLEhessian',hessian);
[~,res] = LL_ConQMLE(Y,thetaest,p,j);
Sigmahat = res'*res/Teff;

% calculate criterion value 
ldSigmahat = log(det(Sigmahat));
% calculate LogLikelihood
LL_est = -0.5*(Teff*ldSigmahat+trace(res/(Sigmahat)*res'));
%LL = (T-L)*log(det(Sigmahat))+(T-L)*K;

% Calculate parameters 
parNu = p*(j*(M^2+N^2)-j^2);
%parNu = L*(p*(M^2+N^2)-((p*(p-1))/2)); % this is no count on othogonal

%AIC: Akaike Information Criterion
ic_Aic = ldSigmahat + parNu*(2/T);

%AIC: Corrected Akaike Information Criterion
ic_Caic = ic_Aic + 2*p*(p+1)/(T-p-1);

%BIC: Schwarz - Bayesian Information Criterion 
ic_SBic = ldSigmahat + parNu*(log(T)/T);

%HQC : Hannan-Quinn Criterion
ic_HQic = ldSigmahat + parNu*(2*log(log(T))/T);
%

% calculate coefficient matrices 
[Aest,Best] = param2MaTS(thetaest,M,N,p,j);

% VAR coefficients such that, Phi = kron(B,A)
Phi = zeros(K,K,p); 
for l=1:p
    for jr=1:j
        Phi(:,:,l) = Phi(:,:,l) + kron(Best(:,:,l,jr),Aest(:,:,l,jr));
    end
end

[~,ComMatrix] = VARCompMatrix(Phi);
EigenValCM = abs(eig(ComMatrix));
LambdaCM = max(EigenValCM);
StabilityCheck = lt(LambdaCM,1);
%SigmaRow = []; % Not important this is just for output consistency 
%SigmaColumn = []; % Not important this is just for output consistency 
%Est_Resuts = {Est_info,,SigmaRow,SigmaColumn,ic_Aic,ic_SBic,ic_HQic,ic_Caic,parNu};
MaTSEst_Results = struct('EstInfo',MaTSEst_info,'CodeName','ConstQMLE_MaTS.m','LoglikelihoodValue',LL_est,...
    'SigmaHat',Sigmahat,'LogDetSigmaHat',ldSigmahat,'Ahat',Aest,'Bhat',Best,'VARCoefficients',Phi,'Residuals',res,...
    'MatsAIC',ic_Aic,'MatsAICc',ic_Caic,'MatsSBIC',ic_SBic,'MatsHQC',ic_HQic,'SampleSizeT',T,'RowDimension',M,...
    'ColumnDimension',N,'NumberOfLags',p,'NumberOfTerms',j,'Df',parNu,'Stability',StabilityCheck);
Est_Resuts_Table = 0 ;
if plots
    figure;
    hist(res,20);
    title('Residuals multi term MAR_j(p)')

end
if printresult
    fprintf("q-MLE #%d (of max %d) \n",it,maxit);
end
end