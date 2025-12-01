%% Alternating Least Squares Estimation for Matrix Time Series
function [MaTSEst_Results,Est_Results_Table,Est_details_output] = alseMARjp(Y1,jlist,plots,printlevel,tol)
%
% alseMARjp Function to estimate multi-term MAR(p), MARj(p) 
%
% Syntax: 
% 
%   [MaTSEst_Results] = alseMARjp(Y,[3 2 1]);
%
% -------------------------------------------------------------------------
% Description:
%
%   alseMARjp estimates multi-term MARj(p) model by alternating least
%       squares. Minimizes sum of squares error given model:
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
%   Est_Results_Table - Table of output. prints Not defined yet.
%
%   Est_details_output - Details of output. prints Not defined yet.
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
    printlevel = 0;
end
if nargin<3
    plots = 0;
end
if nargin<2
    jlist = 1;
end  
%
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
[T,M,N] = size(Y1);
p = size(jlist,2);       % Number of lags
jmax = max(jlist);       % Maximum number of terms
jsum = sum(jlist);       % Sum of terms
K = M*N;                 % Number of variables in vectorized model.
Teff = T-p;              % Effective sample size
%yv = reshape(Y1,T,K); 
Y = permute(Y1,[2,3,1]); % tranforms to M by N by T array 
%----------------------------
% INITIALIZION
%----------------------------
% initialize randomly
% Ainibaset = randn(M,M,p);
% Binibase = randn(N,N,p);
%
%Initial Estimation
%
[Ainibase,Binibase] = initialMaTS(Y1,p,jmax);
%
MaTSEst_info = struct('EstimationMethod','Alternatimg Least Squares');
MaTSEst_info.CodeName = 'alseMARjp.m';
MaTSEst_info.SampleSizeT = T;
MaTSEst_info.RowDimension = M;
MaTSEst_info.ColumnDimension = N;
MaTSEst_info.NumberOfLags = p;
MaTSEst_info.NumberOfTerms = jlist;
MaTSEst_info.RandomNuGen = ranugeMultiMaTSest;
MaTSEst_info.InitialA = Ainibase;
MaTSEst_info.InitialB = Binibase;
%MaTSEst_info.EstimatedData = Y;
MaTSEst_info.readme = 'Output: Est_Resuts,Est_Resuts_Table,Est_details_output.'; 
% Make cell for A and B matrix pairs
ABini = cell(p,2);
% 
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
Yt = Y(:,:,(p+1:end)); 
% Construct MAR(p)
Yt_2 = zeros(M*p,N*p,(Teff));
for ji = 1:p
    Yt_2((M*(ji-1))+1:M*ji,(N*(ji-1))+1:N*ji,:) = Y(:,:,(1+(p-ji)):(end-ji));
end
% Build X(t) pages: (M*Jsum) x (N*Jsum) x (T-p)
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
% ---------- Initial SSE ----------
Yhat = pagemtimes((pagemtimes(Aest,'none',Yt_1,'none')),'none',Best,'transpose');
Rt = Yt - Yhat;
err0 = (norm(Rt,"fro"))^2;
sse = err0;
SSEhistory = sse;
vres0 = reshape(permute(Rt,[3,1,2]),Teff,K);
Sigmahat0 = vres0'*vres0/Teff;
ldSigmahat0 = log(det(Sigmahat0));
ldSigmahat = ldSigmahat0;
%
iter = 0;
if printlevel>0
    FunctionValue = err0;
    fprintf('===================================================================================\n');
    fprintf("   Iteration        f(x)          Improvement     Step Size\n");
    fprintf("      %d             %f \n",iter,FunctionValue);
end
%
% -----------------------------------
% iteration
% -----------------------------------
%
EstPairs = ABini;
diff_syst =T*M*N;
change = 1;
converged = false;
%
while ~converged     
    % update 
    iter = iter+1;
    Aold = Aest;
    Bold = Best;
    EstPairsOld = EstPairs;
    sse_old = sse;
    diff_syst_old = diff_syst;
    ldSigmahatOld = ldSigmahat;

    % estimate A for given B
    parta1 = pagemtimes(Best,'none',Yt_1,'transpose');
    parta = pagemtimes(Yt,'none',parta1,'none');
    Sumparta = sum(parta,3);

    parta2 = pagemtimes(Yt_1,'none',Best,'transpose');
    partaa = pagemtimes(parta2,'none',parta1,'none');
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
    % estimate B(1,1) for given A and SigmaR
    
    partb1 = pagemtimes(Aest,'none',Yt_1,'none');
    partb = pagemtimes(Yt,'transpose',partb1,'none');
    Sumpartb = sum(partb,3);

    partb2 = pagemtimes(Yt_1,'transpose',Aest,'transpose');
    partbb = pagemtimes(partb2,'none',partb1,'none');
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

    % Residuals
    Yhat = pagemtimes((pagemtimes(Aest,'none',Yt_1,'none')),'none',Best,'transpose');
    Rt = Yt - Yhat;
    sse = (norm(Rt,"fro"))^2;
    SSEhistory(end+1) = sse;
    vres = reshape(permute(Rt,[3,1,2]),Teff,K);
    Sigmahat = vres'*vres/Teff;
    ldSigmahat = log(det(Sigmahat));
    StepSzSigma = (ldSigmahatOld - ldSigmahat)/ldSigmahatOld;
    %
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
    aNORM = 0;
    bNORM = 0;
    for li = 1:p
        Ax = cell2mat(EstPairs(li,1));
        AxOld = cell2mat(EstPairsOld (li,1));
        %aNORM = aNORM +sum(pagenorm(Ax-AxOld,"fro")>0.00001);
        aNORM = aNORM + sum(pagenorm(Ax-AxOld,"fro") / max(1, norm(AxOld,'fro')));
        Bx = cell2mat(EstPairs(li,2));
        BxOld = cell2mat(EstPairsOld (li,2));
        %bNORM = bNORM +sum(pagenorm(Bx-BxOld,"fro")>0.00001);
        bNORM = bNORM + sum(pagenorm(Bx-BxOld,"fro") / max(1, norm(BxOld,'fro')));
    end
    diff_syst = aNORM + bNORM;
    StepSz_syst = abs(diff_syst_old - diff_syst) / max(1,abs(diff_syst_old));
    %
    if plots>1
        % calculate LogLikelihood
        LL_est = -0.5*(Teff*ldSigmahat+trace(vres/(Sigmahat)*vres'));
        %
        set(0, 'CurrentFigure', pict1)
        hold on
        grid on
        scatter(iter,LL_est,'filled');
        %
        set(0, 'CurrentFigure', pict2)
        hold on
        grid on
        plot(iter,ldSigmahat,'b*');
        %
        norm_estYhat = norm(Yhat,"fro");
        norm_Yt = norm(Yt,"fro");
        set(0, 'CurrentFigure', pict3)
        hold on
        grid on
        plot(iter,norm_estYhat,'r*');
        scatter(iter,norm_Yt,'filled');
    end
    %
    Improvment = (sse_old - sse)/(sse_old);
    %
    % print out results, if wanted 
    if printlevel>0
        FunctionValue = sse;    
        %fprintf("   Iteration        f(x)          Improvement     Step Size\n");
        fprintf("      %d             %f      %f        %f   %f   %f   %f\n",...
            iter, FunctionValue, Improvment, diff_syst, StepSz_syst, ldSigmahat,StepSzSigma);
    end
    %
    % ------ Stopping rules ------
    % (1) coefficients convergence
    coef_conv = max(aNORM, bNORM) < tol;

    % (2) SSE improvement (non-increasing + small improvement threshold)
    sse_improved = (sse_old - sse) >= tol;

    if coef_conv 
        converged = true;
        break;
    end

    if ~sse_improved
        % stop when we cannot further reduce SSE (within TolSSE); keep the better of (old vs new)
        if sse > sse_old
            % rollback to previous (strictly better) iterate
            %EstPairs = EstPairsOld;
            Aest = Aold ;
            Best = Bold;
            Yhat = pagemtimes((pagemtimes(Aest,'none',Yt_1,'none')),'none',Best,'transpose');
            Rt = Yt - Yhat;
            %sse = (norm(Rt,"fro"))^2;
            vres = reshape(permute(Rt,[3,1,2]),Teff,K);
            Sigmahat = vres'*vres/Teff;
            ldSigmahat = log(det(Sigmahat));
            SSEhistory(end) = sse_old;
        end
        break;
    end
    % MaxSafe: Additinal Safe stop for infinite iteration 
    if (Improvment < 1.0e-10) || (isinf(ldSigmahat))
        break
    end
    if ((abs(StepSz_syst) < 1.0e-10) || iter>10000)
        break
    end

end
if printlevel>0
    fprintf('===================================================================================\n');
    fprintf('===================================================================================\n');
end
 
% calculate LogLikelihood without constants
LL_est = -0.5*(Teff*ldSigmahat+trace(vres/(Sigmahat)*vres'));
%LL = -0.5*(Teff*ldSigmahat+Teff*K);
% Calculate parameter Lag by Lag
NuofPar = zeros(1,p);
for ii = 1:p
    NuofPar(ii)=(jlist(ii)*(M^2+N^2)-jlist(ii)^2);
end
parNu = sum(NuofPar);
%parNu = L*(p*(M^2+N^2)-((p*(p-1))/2));
    %-((p*(p-1))/2));
%AIC: Akaike Information Criterion
ic_Aic = ldSigmahat + parNu*(2/T);

%AIC: Corrected Akaike Information Criterion
ic_Caic = ic_Aic + 2*p*(p+1)/(T-p-1);
%ic_Caic = ic_Aic + (2*parNu)*(parNu+1)/T-parNu-1;

%BIC: Schwarz - Bayesian Information Criterion 
ic_SBic = ldSigmahat + parNu*(log(T)/T);
%ic_Caic = ic_SBic + parNu;

%HQC : Hannan-Quinn Criterion
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
%
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
MaTSEst_Results = struct('EstInfo',MaTSEst_info,'CodeName','alseMARjp.m','LoglikelihoodValue',LL_est,...
    'SigmaHat',Sigmahat,'LogDetSigmaHat',ldSigmahat,'Ahat',Aestarray,'Bhat',Bestarray,'VARCoefficients',Phi,'Residuals',vres,...
    'MatsAIC',ic_Aic,'MatsAICc',ic_Caic,'MatsSBIC',ic_SBic,'MatsHQC',ic_HQic,'SampleSizeT',T,'RowDimension',M,...
    'ColumnDimension',N,'NumberOfLags',p,'NumberOfTerms',jlist,'Df',parNu,'Stability',StabilityCheck);
MaTSEst_Results.converged = converged;
MaTSEst_Results.SSE = SSEhistory;
Est_Results_Table = 0;
Est_details_output = struct('Iteration',iter,'MinPossible',change,'BreakReason',abs(StepSzSigma));
if plots>0
    figure;
    hist(vres,20);
    title('Residuals multi term MAR_j(p)')

end
end