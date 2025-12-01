%% Alternating Maximum Likehood Estimation for Matrix Time Series
function [MaTSEst_Results,Est_Results_Table,Est_details_output] = mleMARjp(Y1,jlist,plots,printlevel,tol)
%
% mleMARjp Function to estimate multi-term MAR(p), MARj(p) 
%
% Syntax: 
% 
%   [MaTSEst_Results] = mleMARjp(Y,[3 2 1]);
%
% -------------------------------------------------------------------------
% Description:
%
%   alseMARjp estimates multi-term MARj(p) model by alternating maximum
%   likelihood under matrix normal distribution given model:
%
%           Y(t) = A_{1,1}*Y(t-1)*B_{1,1}'+ A_{1,2}*Y(t-1)*B_{1,2}'+ ... 
%
%                  + A_{2,1}*Y_(t-2)*B_{2,1}'+ A_{2,2}*Y(t-2)*B_{2,2}' + ... 
%                                               
%                     + A_{p,j}*Y_(t-p)*B_{p,j}' + E_t
%
% -------------------------------------------------------------------------
% Input Arguments:
%
%   Y1 - Data representing observations of a multi-dimensional time series
%       Y(t), specified as a three dimensional array of dimension T x M x N. 
%       T: Time dimension, R: Row dimension, M: Column dimension.
%       The last observation is the most recent. 
%
%   jlist - number of terms per lag, vector of integers, i.e.: [1 2 3]
%
%   plots - integer; 1 plots residuals, 2 plots iteration progress.
%
%   printlevel - integer; indicating, what information should be
%                          provided in the iterations,0,1,2.
%
%   tol - real; tolerance for subsequent iterations.
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
%       converged -
%       SSE - Sum of Squared Errors of iterations.
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
if nargin<5
    tol = 1.0e-6;
end
if nargin<4
    printlevel = 0; %1.0e-4; %1e-6; % 0.000000001;
end
if nargin<3
    plots = 0;
end
if nargin<2
    jlist = 1;
end
if plots>1
    pict1 = figure;
    title('Log Likelihood Value')
    pict2 = figure;
    title('Log(Det(Sigma))')
    pict3 = figure;
    title('Estimated System Value')
end
%
ranugeMultiMaTSest = rng("shuffle");
p = size(jlist,2);      % Number of lags
jmax = max(jlist);      % Maximum number of terms
jsum = sum(jlist);      % Sum of terms
%
[T,M,N] = size(Y1);
K = M*N;                % Number of variables in vectorized model.
Teff = T-p;             % Effective sample size
%vy = reshape(Y1,T,K);
Y = permute(Y1,[2,3,1]);% tranforms to M by N by T array
%----------------------------
% INITIALIZION
%----------------------------
% initialize randomly
% Ainibaset = randn(M,M,p);
% Binibase = randn(N,N,p);
% sigmaR = eye(M);
% sigmaC = eye(N);
%
%Initial Estimation
%
[Ainibase,Binibase,sigmaR,sigmaC] = initialMaTS(Y1,p,jmax);
%
MaTSEst_info = struct('EstimationMethod','Maximum Likelihood Estimation','CodeName','MLE_MARdiffJSforMS_Mark3.m','SampleSizeT',T,...
    'RowDimension',M,'ColumnDimension',N,'NumberOfLags',p,'NumberOfTerms',jlist,...
    'RandomNuGen',ranugeMultiMaTSest);
MaTSEst_info.InitialA = Ainibase;
MaTSEst_info.InitialB = Binibase;
%MaTSEst_info.EstimatedData = Y;
MaTSEst_info.readme = 'Output: Est_Resuts,Est_Resuts_Table,Est_details_output.'; 
% Make cell for A and B matrix pairs
ABini = cell(p,2);
for li = 1:p
    ABini(li,1) = {squeeze(Ainibase(:,:,li,1:jlist(li)))};
    ABini(li,2) = {squeeze(Binibase(:,:,li,1:jlist(li)))};
end
% Arrange initial matrices according to term numbers per lag 
% jlist: i.e.:[1 2 3]
Aest = [];
Best = [];
for li = 1:p
    Aest = [Aest reshape(ABini{li,1},M,M*jlist(li))];
    Best = [Best reshape(ABini{li,2},N,N*jlist(li))];
end
% Build Y pages we use (t = p+1..T) with effective sample size
Yt = Y(:,:,(p+1:end)); % effective sample size
% Construct MAR(p)
Yt_2 = zeros(M*p,N*p,(Teff));
for ji = 1:p
    Yt_2((M*(ji-1))+1:M*ji,(N*(ji-1))+1:N*ji,:) = Y(:,:,(1+(p-ji)):(end-ji));
end

if jmax>1
    Ytemp= [];
    for ji = 1:p
        Ytemp = [Ytemp repmat(Yt_2((M*(ji-1))+1:M*ji,(N*(ji-1))+1:N*ji,:),1,jlist(ji))]; 
    end
    Ytcell = mat2cell(Ytemp, M, ones(1,sum(jlist))* N, Teff);
    Yt_1 = zeros(M*jsum,N*jsum,Teff);
    for ji = 1:jsum
        Yt_1((M*(ji-1))+1:M*ji,(N*(ji-1))+1:N*ji,:) = Ytcell{ji};
    end
else
    Yt_1 = Yt_2;
end
%
% ---------- Initial Loglikelihood ----------
Yhat = pagemtimes((pagemtimes(Aest,'none',Yt_1,'none')),'none',Best,'transpose');
Rt = Yt - Yhat;
err0 = (norm(Rt,"fro"))^2;
err = err0;
vres = reshape(permute(Rt,[3,1,2]),Teff,K);
omegay1V = vres'*vres/Teff;
ldSigmahat0 = log(det(omegay1V));
ldSigmahat = ldSigmahat0;
LL_est0 = -0.5*T*ldSigmahat + +Teff*K;
LL_est = LL_est0;
%
if printlevel>0
    FunctionValue = LL_est0;
    fprintf('===================================================================================\n');
    fprintf("   Iteration        f(x)          Improvement     Step Size\n");
    fprintf("      %d             %f \n",0,FunctionValue);
end
%
% -----------------------------------
% iteration
% -----------------------------------
%
it = 0;
change = 1;
EstPairs = ABini;
diff_syst =T*M*N;
while (change>tol)
    % update 
    Aold = Aest;
    Bold = Best;
    EstPairsOld = EstPairs;
    sigmaRold = sigmaR;
    sigmaCold = sigmaC;
    err_old = err;
    diff_syst_old = diff_syst;
    LLest_old = LL_est;
    ldSigmahatOld = ldSigmahat;

    % estimate B for given A and SigmaR
    TY = pagetranspose(Yt); 
    partb1 = pagemrdivide(TY,sigmaR);
    partb2 = pagemtimes(Aest,'none',Yt_1,'none');
    partb = pagemtimes(partb1,'none',partb2,'none');
    Sumpartb = sum(partb,3);

    partb3 = pagemtimes(Yt_1,'transpose',Aest,'transpose');
    partb4 = pagemrdivide(partb3,sigmaR);
    partbb = pagemtimes(partb4,'none',partb2,'none');
    Sumpartbb = sum(partbb,3);

    Best = Sumpartb/Sumpartbb;

    % Normalization with frobenous norm of ||Bi||_F=1, this is optional to
    % generic norm=1 normalisation above, in addition it can be added...
%{
    Bestarray = reshape(Best,N,N,L,p);
    pnBest = pagenorm(Bestarray,"fro");
    Bnor = pagemrdivide(Bestarray,pnBest);
    Best = reshape(Bnor,N,N*L*p);
%}
        % estimate A for given B

    parta1 = pagemrdivide(Yt,sigmaC);
    parta2 = pagemtimes(Best,'none',Yt_1,'transpose');
    parta = pagemtimes(parta1,'none',parta2,'none');
    Sumparta = sum(parta,3);

    parta3 = pagemtimes(Yt_1,'none',Best,'transpose');
    parta4 = pagemrdivide(parta3,sigmaC);
    partaa = pagemtimes(parta4,'none',parta2,'none');
    Sumpartaa = sum(partaa,3);

    Aest = Sumparta/Sumpartaa;
    
    % Normalization
    % This is generic norm=1 normalisation such that ||Ai,r||=1, i=1,..,p
    % r=1,...,Jp
%{   
    Aestarray = reshape(Aest,M,M,psum);
    pnAest = pagenorm(Aestarray,"fro");
    Anor = pagemrdivide(Aestarray,pnAest);
    Aest = reshape(Anor,M,M*psum);
%}
    %Aest = Aest/ norm(Aest,"fro");

    % Normalization with QR, when using this identification, please do NOT
    % forget to delete/ignore other normaizations above!!
%
    Aesttemp = Aest;
    Besttemp = Best;
    EstPairs = cell(p,2);
    for li = 1:p
        EstPairs(li,1) = {reshape(Aesttemp(:,1:M*jlist(li)),M,M,jlist(li))};
        Aesttemp = Aesttemp(:,M*jlist(li)+1:end); 
        EstPairs(li,2) = {reshape(Besttemp(:,1:N*jlist(li)),N,N,jlist(li))};
        Besttemp = Besttemp(:,N*jlist(li)+1:end); 
    end
    Aest = [];
    Best = [];
    for li = 1:p
        [Anor,Bnor] = normalizeAB(cell2mat(EstPairs(li,1)),cell2mat(EstPairs(li,2))); 
        Aest = [Aest reshape(Anor,M,M*jlist(li))];
        Best = [Best reshape(Bnor,N,N*jlist(li))];
    end
%
    % Residuals
    estYhat = pagemtimes((pagemtimes(Aest,'none',Yt_1,'none')),'none',Best,'transpose');
    Rt = Yt - estYhat;
    TRt = pagetranspose(Rt);
    % Estimmate sigmas
    % First SigmaR
    
    partsr1 = pagemrdivide(Rt,sigmaC);
    sigmaR = sum((pagemtimes(partsr1,'none',TRt,'none')),3)/(N*Teff);
    
    %Normalization
    % Normalization with frobenous norm such that ||sigmaR ||=1
%    sigmaR = sigmaR/ norm(sigmaR,"fro");

    % SigmaC
    partsc1 = pagemrdivide(TRt,sigmaR);
    sigmaC = sum((pagemtimes(partsc1,'none',Rt,'none')),3)/(M*Teff);

    % Normalization with QR
    [sigmaR,sigmaC] = normalizeAB(sigmaR,sigmaC);

    Aesttemp = Aest;
    Besttemp = Best;
    EstPairs = cell(p,2);
    for li = 1:p
        EstPairs(li,1) = {reshape(Aesttemp(:,1:M*jlist(li)),M,M,jlist(li))};
        Aesttemp = Aesttemp(:,M*jlist(li)+1:end); 
        EstPairs(li,2) = {reshape(Besttemp(:,1:N*jlist(li)),N,N,jlist(li))};
        Besttemp = Besttemp(:,N*jlist(li)+1:end); 
    end
    aNORM = 0;
    bNORM = 0;
    for li = 1:p
        Ax = cell2mat(EstPairs(li,1));
        AxOld = cell2mat(EstPairsOld (li,1));
        %aNORM = aNORM +sum(pagenorm(Ax-AxOld,"fro")>0.00001);
        aNORM = aNORM + sum(pagenorm(Ax-AxOld,"fro"));
        Bx = cell2mat(EstPairs(li,2));
        BxOld = cell2mat(EstPairsOld (li,2));
        %bNORM = bNORM +sum(pagenorm(Bx-BxOld,"fro")>0.00001);
        bNORM = bNORM +sum(pagenorm(Bx-BxOld,"fro"));
    end
    %
    diff_syst = aNORM + bNORM + norm(sigmaR-sigmaRold,"fro") + norm(sigmaC-sigmaCold,"fro");
    StepSz_syst = (diff_syst_old - diff_syst)/diff_syst_old;
    %    
    LL_est = -0.5*(N*Teff*log(det(sigmaR))+M*Teff*log(det(sigmaC))+Teff*K);
    errLL = (LL_est - LLest_old);
    change = max(errLL,diff_syst);
    err = norm(Rt,"fro");
    %
    Sigmahat = kron(sigmaC,sigmaR); 
    ldSigmahat = log(det(Sigmahat));
    StepSzSigma = (ldSigmahatOld - ldSigmahat)/ldSigmahatOld;
    %
    if plots>1
        set(0, 'CurrentFigure', pict1)
        title('LL Value')
        grid on
        hold on
        scatter(it,LL_est,'filled');
        %
        set(0, 'CurrentFigure', pict2)
        title('ldSigmahat')
        grid on
        hold on
        plot(it,ldSigmahat,'b*');
        %
        norm_estYhat = norm(estYhat,"fro");
        norm_Yt = norm(Yt,"fro");
        set(0, 'CurrentFigure', pict3)
        title('Error')
        hold on
        grid on
        plot(it,norm_estYhat,'r*');
        scatter(it,norm_Yt,'filled');
    end
    %
    Improvment = (LLest_old - LL_est)/LLest_old;
    it = it+1;
    % print out results, if wanted 
    if printlevel>0
        %FunctionValue = err^2/(T*M*N);
        FunctionValue = LL_est;
        %fprintf("   Iteration        f(x)          Improvement     Step Size\n");
        fprintf("      %d             %f      %f        %f   %f   %f   %f\n",...
            it, FunctionValue, Improvment, diff_syst, StepSz_syst, ldSigmahat,StepSzSigma);
    end
    %
    if it > 1
        if (Improvment < 1.0e-6) || (isinf(ldSigmahat))
            %(abs(StepSzSigma) < tol) || (isinf(ldSigmahat))
            %(abs(StepSzSigma) < tol) || (isinf(ldSigmahat))
            break;
        end
    end

   % LL_improved = (LL_old - LL_est) >= 1.0e-8;
    if ((abs(StepSz_syst) < 1.0e-6) || it>1000)
        break;
    end
end
if printlevel>0
    fprintf('===================================================================================\n');
    fprintf('===================================================================================\n');
end
vres = reshape(permute(Rt,[3,1,2]),Teff,K);
Sigmahat2 = vres'*vres/Teff;
Sigmahat = kron(sigmaC,sigmaR);

%checksigma = isequal(Sigmahat,Sigmahat2)
norm(Sigmahat-Sigmahat2,"fro");
% calculate criterion value 
ldSigmahat = log(det(Sigmahat));
%ldSigmahat2 = log(det(Sigmahat2));

% Calculate parameter Lag by Lag
NuofPar = zeros(1,p);
for ii = 1:p
    NuofPar(ii)=(jlist(ii)*(M^2+N^2)-jlist(ii)^2);
end
parNu = sum(NuofPar);
%AIC: Akaike Information Criterion
ic_Aic = ldSigmahat + parNu*(2/T);

%AIC: Corrected Akaike Information Criterion
ic_Caic = ic_Aic + (2*p)*(p+1)/(T-p-1);

%BIC: Schwarz - Bayesian Information Criterion
ic_SBic = ldSigmahat + parNu*(log(T)/T);

%HQC: Hannan-Quinn Criterion
ic_HQic = ldSigmahat + parNu*(2*log(log(T))/T);

% VAR coefficients such that, Phi = kron(B,A)
Aesttemp = reshape(Aest,M,M,jsum);
Besttemp = reshape(Best,N,N,jsum);
Aestarray = zeros(M,M,p,jmax);
Bestarray = zeros(N,N,p,jmax);
for ii = 1:p
    Aestarray(:,:,ii,1:jlist(ii)) = Aesttemp(:,:,1:jlist(ii));
    Aesttemp = Aesttemp(:,:,jlist(ii)+1:end);
    Bestarray(:,:,ii,1:jlist(ii)) = Besttemp(:,:,1:jlist(ii));
    Besttemp = Besttemp(:,:,jlist(ii)+1:end);
end
% VAR coefficients such that, Phi = kron(B,A)
Phi = zeros(K,K,p); 
for li=1:p
    for jr=1:jmax
        Phi(:,:,li) = Phi(:,:,li) + kron(Bestarray(:,:,li,jr),Aestarray(:,:,li,jr));
    end
end
[~, ComMatrix] = VARCompMatrix(Phi);
EigenValCM = abs(eig(ComMatrix));
LambdaCM = max(EigenValCM);
StabilityCheck = lt(LambdaCM,1);
MaTSEst_Results = struct('EstInfo',MaTSEst_info,'CodeName','MLE_MARdiffJSforMS_Mark3.m','LoglikelihoodValue',LL_est,...
    'SigmaHat',Sigmahat,'LogDetSigmaHat',ldSigmahat,'SigmaHatRow',sigmaR,'SigmaHatColumn',sigmaC,'Ahat',Aestarray,...
    'Bhat',Bestarray,'VARCoefficients',Phi,'Residuals',vres,...
    'MatsAIC',ic_Aic,'MatsAICc',ic_Caic,'MatsSBIC',ic_SBic,'MatsHQC',ic_HQic,'SampleSizeT',T,'RowDimension',M,...
    'ColumnDimension',N,'NumberOfLags',p,'NumberOfTerms',jlist,'Df',parNu,'Stability',StabilityCheck);
Est_Results_Table = 0;
Est_details_output = struct('Iteration',it,'MinPossible',change,'BreakReason',abs(StepSzSigma));
if plots>0
    figure;
    hist(vres,20);
    title('Residuals multi term MAR_j(p)')

end
end