%% Alternating estimation Matrix ECM with modified Johansen's method
% performs cointegration test 
%
function [ECMaTSEst_Results,ECEst_Results_Table,ECEst_details_output] = vECMAR(Y,q,alpha_crit,Johcases,plotsfigures,printlevel,tol)
%
% vECMAR   Function to estimate multi-term MAR(p), MARj(p) model in 
%               ingerated order one, I(1) cointegrated process.
%
% Syntax:
%
%   out = vECMAR(Y,q,alpha_crit,Johcases,plotsfigures,printlevel,tol);
%
% ------------------------------------------------------------
% Description:
%
%   vECMAR     Function estimates Cointegrated Matrix 
%                    Autoregresive Time Series 
% -------------------------------------------------------------------------
% Input Arguments:
%
%   Y1 - Data representing observations of a multi-dimensional time series
%       Y(t), specified as a three dimensional array of dimension T x M x N. 
%       T: Time dimension, R: Row dimension, M: Column dimension.
%       The last observation is the most recent. 
%
%   q - number of terms per lag, vector of integers, i.e.: [1 2 3]
%   
%   alpha_crit - Significance levels for the trace tests. Values can be 
%                   between 0.10 and 0.01. The default value is 0.05.
%   
%   Johcases - Johansen 5 cases for models. No deterministic variables 
%                   included in this version!
%
%   plotsfigures - integer; 1 plots residuals, 2 plots iteration progress.
%
%   printlevel - integer; indicating, what information should be
%                          provided in the iterations,0,1,2.
%   
%   tol - real; tolerance for subsequent iterations.
%
% -------------------------------------------------------------------------
% Output Arguments:
%
%    ECMaTSEst_Results - Results for estimated vECMARj(p) model. It is a 
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
%   ECMaTSEst_Results_Table - Table of output. prints Not defined yet.
%
%   ECMaTSEst_Results_details_output - Details of output. prints Not defined yet.
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
% Date:         01.08.2025
% written by:   Kurtulus Kidik 
%               Econometrics
%               Universität Bielefeld
%               Universitätsstraße 25
%               D-33615 Bielefeld
%               kurtulus.kidik@uni-bielefeld.de
% -------------------------------------------------------------------------
if nargin<7
    tol = 1.0e-6;%1.0e-10;
end
if nargin<6
    printlevel = 0;
end
if nargin<5
    plotsfigures = 0;
end
if nargin<4
    Johcases = 5;
end
if nargin<3
    alpha_crit = 0.05;
end
if nargin<2
    q = 1;
end

if plotsfigures>1
    pict1 = figure;
    title('Log Likelihood Value')
    pict2 = figure;
    title('Log(Det(Sigma))')
    pict3 = figure;
    title('Estimated System Value')
end
[T,M,N] = size(Y);
K = M*N;
jlist = sort(1:q,'descend'); % this is general term structure
%jlist = [6 12];
%minj = min(M*M,N*N);
%jlist = [minj minj];
Jmax = max(jlist);
jsum = sum(jlist);

Teff = T-(q+1);

if N <= 1
    error('!!! Attention !!! :Y is Not three dimensional array data !!!, please press any key to continue !!!')
end

if Jmax>min(M*M,N*N)
    sprintf('!!! Attention !!! Number of Terms can NOT be bigger than minimum of square of dimensions... !!!!!!')
    sprintf('Choose Max. term=: %d',min(M*M,N*N))
end
ranugeECMaTSest = rng;
ECMaTSEst_info = struct('EstimationMethod','ECMARwithJoh','CodeName','vECMAR.m','SampleSizeT',T,...
    'RowDimension',M,'ColumnDimension',N,'NumberOfLags',q,'NumberOfTerms',jlist,...
    'RandomNuGen',ranugeECMaTSest);
%MaTSEst_info.InitialA = Ainibase;
%MaTSEst_info.InitialB = Binibase;
%ECMaTSEst_info.EstimatedData = Y;
ECMaTSEst_info.readme = 'Output: Est_Resuts,Est_Resuts_Table,Est_details_output.'; 

y = reshape(Y,T,K);          %y: Vectorized Y (T by M*N)

% Generate Differences, Lagged Level and Lagged Differences
y_1 = y(1:end-1,:);                   % Vectorized Y_1 (T-1 by M*N)
dy = diff(y,1,1);                     % Differenced y (T-1 by M*N)
DeltaY_t = diff(Y,1,1);               %DeltaY_t. (T-1 by M by N)
dy_test = reshape(DeltaY_t,T-1,M*N);  %Deltay_t: Vectorized DeltaY_t (T-1 by M*N)
%Z1t_y = y_1((L+1):end,:);
       
if dy ~= dy_test
  error('!!! Attention !!! : !!!!!!')
%  stop;  % Wait for user to type any key
end

err0 = norm(DeltaY_t,"fro");
if printlevel>0
    FunctionValue0 = err0^2/(T*K);
    fprintf('===================================================================================\n');
    fprintf("   Iteration    rank    f(x)          error          Improvement     Syst. Diff      StepSz_syst      LDSigma      StepSz_Sigma\n");
    fprintf("      %d        %d     %f        %f \n",-1,0,err0,FunctionValue0);
end

%Z0t = DeltaY_t((L+1):end,:,:);
Z0t = DeltaY_t;
Z1t = Y((q+1):end-1,:,:);
Z2t = DeltaY_t((q+1):end,:,:); % Not exactly as definition
Z2vt = reshape(Z2t,Teff,K);    % Not exactly as definition, this effective delta part
Z1vt = reshape(Z1t,Teff,K);


it = 0;
Pi_old = zeros(K);
P_betaest_old = 0;
est_err_beta_old = 0;
diff_syst_old = 0;
ldSigmahat0 = log(det((Z2vt'*Z2vt)/Teff));
R1t = Z1vt;
Stability = 0;
itst =0;
while Stability ~=1
    if itst <= M*N
        MaTSEst_Results = initialECMAR(Y,jlist,itst); % initial estimation
    elseif itst == M*N+1
        Pimatrix = (R1t\Z2vt)';
        xt1 = Z2vt - R1t*Pimatrix';
        Xt = (reshape(xt1,Teff,M,N));
        % Estimate Gammas given alpha and beta
        ExoMaTSEst_Results = ALS_MAR_Exo(Xt,Z0t,jlist); %
        pY = permute(Z0t,[2,3,1]);
        shortT = zeros(Teff,K);
        for ii = 1:q
            %LagsTerms = zeros(M,N,Teff);
            for ik = 1:jlist(ii)
                temp = pagemtimes(ExoMaTSEst_Results.Ahat(:,:,ii,ik),'none',pY(:,:,(1+(q-ii)):(end-ii)),'none');
                temp2 = pagemtimes(temp,'none',ExoMaTSEst_Results.Bhat(:,:,ii,ik),'transpose');
                LagsTerms = permute(temp2,[3,1,2]);
                shortT = shortT + reshape(LagsTerms,Teff,K);
            end
        end
        MaTSEst_Results.Residuals = Z2vt - shortT;
    else 
        Stability = 1;
    end
R0t = MaTSEst_Results.Residuals;

size(R0t,1) == size(R1t,1);

% Estimate alpha and beta given Gammas
% Reduced Rank Regression 
RREst_Results = RedRankReg(R0t,R1t,alpha_crit,Johcases); % ***************
%
b_eta1 = RREst_Results.CRank;
alphahat = RREst_Results.AlphaHat;
betahat = RREst_Results.BetaHat;
tracevec1 = RREst_Results.TraceVal;
%
% Initial Estimation finished 
betaest = betahat;
alphaest = alphahat;
% Projection matrix of Beta
P_betaest = (betaest/(betaest'*betaest))*betaest';
est_err_beta1 = log((norm((P_betaest-P_betaest_old),"fro"))^2);
est_err_beta = norm((P_betaest-P_betaest_old),"fro");
PM_error1 = est_err_beta_old - est_err_beta;
PM_error = abs(est_err_beta_old - est_err_beta);
% Projection matrix of Beta --- Ends
%diff_syst = norm(alphaest*betaest'-Pi_old,"fro") + PM_error;
diff_syst1 = norm(alphaest*betaest'-Pi_old,"fro") + est_err_beta;
StepSz_syst1 = diff_syst1;
%StepSz_syst = (diff_syst_old - diff_syst)/diff_syst_old;

xt1  = Z2vt - Z1vt*betaest*alphaest';
Xt = (reshape(xt1,Teff,M,N));
% Estimate Gammas given alpha and beta
ExoMaTSEst_Results0 = ALS_MAR_Exo(Xt,Z0t,jlist); %

pY = permute(Z0t,[2,3,1]);
shortT = zeros(Teff,K);
for ii = 1:q
    %LagsTerms = zeros(M,N,Teff);
    for ik = 1:jlist(ii)
        temp = pagemtimes(ExoMaTSEst_Results0.Ahat(:,:,ii,ik),'none',pY(:,:,(1+(q-ii)):(end-ii)),'none');
        temp2 = pagemtimes(temp,'none',ExoMaTSEst_Results0.Bhat(:,:,ii,ik),'transpose');
        LagsTerms = permute(temp2,[3,1,2]);
        shortT = shortT + reshape(LagsTerms,Teff,K);
    end
end
R0t = Z2vt - shortT;
estYhat  = Z1vt*betaest*alphaest' + shortT;
vRt = Z2vt - estYhat;
verr = norm(vRt,"fro");
Improvment0 = (err0^2 - verr^2)/(err0^2);
Sigmahat = vRt'*vRt/Teff;
ldSigmahat1 = log(det(Sigmahat));
StepSzSigma1 = (ldSigmahat0 - ldSigmahat1)/ldSigmahat0;
FunctionValue1 = verr^2/(T*K);
if printlevel>0
    fprintf("      %d         %d     %f        %f        %f      %f      %f      %f\n",...
        it, b_eta1, FunctionValue1, Improvment0, diff_syst1, StepSz_syst1, ldSigmahat1,StepSzSigma1);

end
%
% estimate Gammas for given alpha, beta
%
    if plotsfigures>1
        % calculate LogLikelihood
        LL_est = -0.5*(Teff*ldSigmahat1+trace(vRt/(Sigmahat)*vRt'));
        %
        set(0, 'CurrentFigure', pict1)
        hold on
        grid on
        scatter(it,LL_est,'filled');
        %
        set(0, 'CurrentFigure', pict2)
        hold on
        grid on
        plot(it,ldSigmahat1,'b*');
        %
        norm_estYhat = norm(estYhat,"fro");
        norm_Yt = norm(Z0t,"fro");
        set(0, 'CurrentFigure', pict3)
        hold on
        grid on
        plot(it,norm_estYhat,'r*');
        scatter(it,norm_Yt,'filled');
        %plot(it,norm(Yt,"fro"),'b*');
 
    end

verr_old = err0;
changerr = verr-verr_old;
change = max(changerr,diff_syst1);
y = 0;
KroBA = zeros(M*N,M*N,q); % transorm to VAR coefficients
    for lt=1:q
        for jr=1:jlist(lt)
            KroBA(:,:,lt) = KroBA(:,:,lt) + kron(ExoMaTSEst_Results0.Bhat(:,:,lt,jr),ExoMaTSEst_Results0.Ahat(:,:,lt,jr));
        end
    end
    
    CompM = zeros(K*(q+1),K*(q+1));
    [~, CompM] = VECM_Coef2VAR_Coef(alphaest, betaest, KroBA);
    lambdaCM = abs(eig(CompM));
    MaxLambda = max(lambdaCM);
    RoundlambdaCM = ceil(lambdaCM*1000)/1000;
    MaxLambdaRounded = max(RoundlambdaCM);
    
    ExoMaTSEst_Results.Residuals = 0;
    Improvment1 = 1;
    change = 1;
    while (change>tol) 
            P_betaest_old = P_betaest;
            est_err_beta_old = est_err_beta;
            Pi_old = alphaest*betaest';
            verr_old = verr;
            ldSigmahat_old = ldSigmahat1;
            diff_syst_old = diff_syst1;
%
            RREst_Results= RedRankReg(R0t,R1t,alpha_crit,Johcases); % *******
%
            b_eta = RREst_Results.CRank;
            alphaest = RREst_Results.AlphaHat;
            betaest = RREst_Results.BetaHat;
            tracevec = RREst_Results.TraceVal;
%
            Pi = alphaest*betaest'; %Improvment1; % StepSz_syst1;
            xt  = Z2vt - Z1vt*Pi';
            Xt = (reshape(xt,Teff,M,N));
            % Estimate Gammas given alpha and beta
            ExoMaTSEst_Results = ALS_MAR_Exo(Xt,Z0t,jlist); %

            % Projection matrix of Beta
            if betaest == 0
                P_betaest = 0;
            else
                P_betaest = (betaest/(betaest'*betaest))*betaest';
            end

            est_err_beta = norm((P_betaest-P_betaest_old),"fro");
            PM_error1 = est_err_beta_old - est_err_beta;
            PM_error = abs(est_err_beta_old - est_err_beta);
            % Projection matrix of Beta --- Ends

            diff_syst1 = norm(alphaest*betaest'-Pi_old,"fro");% + est_err_beta;
            
            StepSz_syst1 = (diff_syst_old - diff_syst1)/diff_syst_old;
   
            pY = permute(Z0t,[2,3,1]);
            shortT = zeros(Teff,K);
            for ii = 1:q
                for ik = 1:jlist(ii)
                    temp = pagemtimes(ExoMaTSEst_Results.Ahat(:,:,ii,ik),'none',pY(:,:,(1+(q-ii)):(end-ii)),'none');
                    temp2 = pagemtimes(temp,'none',ExoMaTSEst_Results.Bhat(:,:,ii,ik),'transpose');
                    LagsTerms = permute(temp2,[3,1,2]);
                    shortT = shortT + reshape(LagsTerms,Teff,K);
                end
            end
            R0t = Z2vt - shortT;
            estYhat  = Z1vt*betaest*alphaest' + shortT;
            vRt = Z2vt - estYhat;
            verr = norm(vRt,"fro");
            changerr = verr-verr_old;
            change = max(changerr,diff_syst1);

            Improvment1 = (verr_old - verr)^2/(verr_old);
            Sigmahat = vRt'*vRt/Teff;
            ldSigmahat1 = log(det(Sigmahat));
            StepSzSigma1 = (ldSigmahat_old - ldSigmahat1)/ldSigmahat_old;
            FunctionValue = verr^2/(T*K);
            it = it+1;
            if printlevel>0
                fprintf("      %d         %d     %f        %f        %f      %f      %f      %f\n",...
                    it, b_eta, FunctionValue, Improvment1, diff_syst1, StepSz_syst1, ldSigmahat1,StepSzSigma1);
            end

                if plotsfigures>1

                    % calculate LogLikelihood
                    LL_est = -0.5*(Teff*ldSigmahat1+trace(vRt/(Sigmahat)*vRt'));
                    %
                    set(0, 'CurrentFigure', pict1)
                    hold on
                    grid on
                    scatter(it,LL_est,'filled');
                    
                    set(0, 'CurrentFigure', pict2)
                    hold on
                    grid on
                    plot(it,ldSigmahat1,'b*');
                    %
                    %
                    norm_estYhat = norm(estYhat,"fro");
                    norm_Yt = norm(Z0t,"fro");
                    set(0, 'CurrentFigure', pict3)
                    hold on
                    grid on
                    plot(it,norm_estYhat,'r*');
                    scatter(it,norm_Yt,'filled');
                    %plot(it,norm(Yt,"fro"),'b*');
             
                end
                if (abs(Improvment1) < 1.0e-10) || (isinf(ldSigmahat1))

                    break  
                end
                if ((abs(StepSz_syst1) < 1.0e-6) || it>500)
                    break

                end
                
    end
    if printlevel>0
        fprintf('===================================================================================\n');
        fprintf('===================================================================================\n');
    end
    KroBA = zeros(M*N,M*N,q); % transorm to VAR coefficients
    for lt=1:q
        for jr=1:jlist(lt)
            KroBA(:,:,lt) = KroBA(:,:,lt) + kron(ExoMaTSEst_Results.Bhat(:,:,lt,jr),ExoMaTSEst_Results.Ahat(:,:,lt,jr));
        end
    end
    
    CompM = zeros(K*(q+1),K*(q+1));
    [~, CompM] = VECM_Coef2VAR_Coef(alphaest, betaest, KroBA);

    lambdaCM = abs(eig(CompM));
    MaxLambda = max(lambdaCM);
    RoundlambdaCM = ceil(lambdaCM*1000)/1000;
    MaxLambdaRounded = max(RoundlambdaCM);

    Stability = MaxLambda <= 1;
    if itst > M*N+1
        break
    end
    itst = itst +1;
end

% calculate LogLikelihood
LL_est = -0.5*(Teff*ldSigmahat1+trace(vRt/(Sigmahat)*vRt'));

% Calculate parameter Lag by Lag
NuofPar = zeros(1,q);
for ii = 1:q
    NuofPar(ii)=(jlist(ii)*(M^2+N^2)-jlist(ii)^2);
end
parNu = sum(NuofPar) + 2*K*b_eta;

%AIC: Akaike Information Criterion
ic_Aic = ldSigmahat1 + parNu*(2/T);

%AIC: Corrected Akaike Information Criterion
ic_Caic = ic_Aic + 2*q*(q+1)/(T-q-1);
%ic_Caic = ic_Aic + (2*parNu)*(parNu+1)/T-parNu-1;

%BIC: Schwarz - Bayesian Information Criterion 
ic_SBic = ldSigmahat1 + parNu*(log(T)/T);
%ic_Caic = ic_SBic + parNu;

%HQC : Hannan-Quinn Criterion
ic_HQic = ldSigmahat1 + parNu*(2*log(log(T))/T);
% VAR coefficients such that, Phi = kron(B,A)

Aestarray = ExoMaTSEst_Results.Ahat;
Bestarray = ExoMaTSEst_Results.Bhat;
Phi = KroBA;

StabilityCheck = Stability;
ECMaTSEst_Results = struct('EstInfo',ECMaTSEst_info,'CodeName','ALS_MARdiffJSforMS_Mark3.m','LoglikelihoodValue',LL_est,...
    'SigmaHat',Sigmahat,'LogDetSigmaHat',ldSigmahat1,'Crank',b_eta,...
    'AlphaHat',alphaest,'BetaHat',betaest,'Ahat',Aestarray,'Bhat',Bestarray,'VARCoefficients',Phi,'Residuals',vRt,...
    'MatsAIC',ic_Aic,'MatsAICc',ic_Caic,'MatsSBIC',ic_SBic,'MatsHQC',ic_HQic,'SampleSizeT',T,'RowDimension',M,...
    'ColumnDimension',N,'NumberOfLags',q,'NumberOfTerms',jlist,'Df',parNu,'Stability',StabilityCheck,...
    'TraceVal',tracevec,'TraceValinit',tracevec1,'initRank',b_eta1);
ECEst_Results_Table = 0;
ECEst_details_output = 0;
%
if plotsfigures>0
    figure;
    hist(vRt,20);
    title('Residuals multi term EC-MAR_j(p)')
    
       
    % Table really corresponding to model without constant
    %load % MAT file contains simulated critical values. 

    load('CV_Johinfinite.mat');
    
    etamat = flipud(CV_Joh.tracecv);
    dim_etamat = size(etamat,1);
    
    % Selection of column corresponding to ALPHA
    if alpha_crit == 0.10
        col = 1;
    elseif alpha_crit == 0.05
        col = 2;
    elseif alpha_crit == 0.025
        col = 3;
    elseif alpha_crit == 0.01
        col = 4;
    end
    figure;
    hold on;
    set(gca,'XTick',0:K-1);
    bar(0:K-1,etamat(dim_etamat-K+1:dim_etamat,col),'b');
    plot(0:K-1,tracevec(1:end),'r*');
    title(sprintf('EC-MAR: Estimated coint rank: %d',b_eta));
end
end
%==================================================================================================================%
 