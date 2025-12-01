function Est_Results = johVECMbase(y,q,alpha_crit,Johcases,plotsRank,printresults);
%
% johVECMbase   Performs Johansen cointegration tests
% Syntax:
%
%   Est_Results = johVECMbase(y,q,alpha_crit,Johcases,plotsRank,printresults);
% ------------------------------------------------------------
% Description:
%
%   johVECMbase     Function for generating data for Cointegrated 
%           Vector Autoregresive Time Series for VECM estimation
% ------------------------------------------------------------
% Input Arguments:
%   
%   y - rank ofPi matrix, alpha, beta 
%   q - lag of delta part
%   alpha_crit - 
%   Johcases - 
%   plotsRank - 
%   printresults - 
%   
% ------------------------------------------------------------
% Output Arguments:
%
%   Est_Results - Results for estimated model. It is a structure with the
%           following fields that correspond directly 
%           to those in the estimated model:
%               
%       Est_Results = struct('EstInfo',RREst_info,'CodeName','johansenCoinTestbase.m');
%       Est_Results.EigVectors = Vmson;     % Normalized Eigenvectors
%       Est_Results.EigValues = lambda;     % Eigenvalues
%       Est_Results.EigIndx = sortIdxDm;    % Index of Eigenvalues (High to low)
%       Est_Results.CRank = b_eta;
%       Est_Results.CRankMeig = b_xi;
%       Est_Results.AlphaHat = alphahat;
%       Est_Results.BetaHat = betahat;
%       Est_Results.SRCoef = Gamma;
%       Est_Results.Residuals = vres;
%       Est_Results.TraceVal = tracevec;
%       Est_Results.MaxEigVal = maxvec;
%       Est_Results.LoglikeValue = LL_est;
%       Est_Results.SigmaHat = Sigma;
%       Est_Results.LogDetSigmaHat = ldSigma;
%       Est_Results.AICValue = ic_Aic;
%       Est_Results.AICcValue = ic_Caic;
%       Est_Results.SBICValue = ic_SBic; 
%       Est_Results.HQCValue = ic_HQic;
%       Est_Results.cValTrace = critmat;
%       Est_Results.cValMeig = maxmat;
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
%   [2] Kidik, K. and Bauer, D. "Specification and Estimation of Matrix 
%       Time Series Models with Multiple Terms". Preprint.
% 
% ------------------------------------------------------------
% Date:         01.08.2025
% written by:   Kurtulus Kidik 
%               Econometrics
%               Universität Bielefeld
%               Universitätsstraße 25
%               D-33615 Bielefeld
%               kurtulus.kidik@uni-bielefeld.de
% ------------------------------------------------------------
if nargin<6
    printresults = 0;
end
if nargin<5
    plotsRank = 0;
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

[T,K] = size(y); % K-dimensional VAR
Teff = T-(q+1);
%
Z0t = diff(y,1,1); % Differenced y (T-1 by K)
Z1t = y(1:end-1,:); % y_1 (T-1 by K)
    
Z2t = zeros(T-1,K*q);
for j=1:q
    Z2t(:,(j-1)*K+[1:K])=[NaN(j,K);Z0t(1:end-j,:)];
end

Z0t = Z0t(q+1:end,:);
Z1t = Z1t(q+1:end,:);
Z2t = Z2t(q+1:end,:);

% Perform the concentrating regressions:
% generate R_0t
R0t = Z0t - Z2t*(Z2t\Z0t); % Z0t on Z2t residuals

% generate R_1t
R1t = Z1t - Z2t*(Z2t\Z1t); % Z1t on Z2t residuals

% Compute the product moment matrices of the residuals:
S11 = (R1t'*R1t)/size(R1t,1);
S00 = (R0t'*R0t)/size(R0t,1);
S10 = (R1t'*R0t)/size(R1t,1);
S01 = (R0t'*R1t)/size(R0t,1);

% Adjust for numerical stability
me1 = min(eig(S11));
if me1<1.0e-05
    mm1 = size(S11,1);
    S11 = S11+eye(mm1)*((1.0e-05)-me1);
end
me2 = min(eig(S00));
if me2< 1.0e-05                    %me2<10^(-4)
    mm2 = size(S00,1);
    S00 = S00+eye(mm2)*((1.0e-05)-me2);
end
%
% Estimate the eigenvectors and eigenvalues of Pi = alpha*beta':
%       
[Vm,Dm] = eig(S10*(S00\S01),S11);
[lambda,sortIdxDm] = sort(diag(Dm),'descend'); % Ordered eigenvalues
Vm = Vm(:,sortIdxDm);                          % Corresponding eigenvectors
                
% Normalize the eigenvectors so that V'*S11*V = I, as in [2]:
        
Um = Vm'*S11*Vm; % V diagonalizes S11
%Vmson1 = bsxfun(@rdivide,Vm,sqrt(diag(Um))');
Vmson = Vm./sqrt(diag(Um))';

    % Test statistics vectors, version with %(T-k) not% T
    switch Johcases %additional vars included at first position due to deterministics? 
        case {2,4}
            loglambda = log(abs(ones(K,1)-lambda(1:end-1,:)));
    
        otherwise
            loglambda = log(abs(ones(K,1)-lambda));   % ABS for rounding lambda ~ 1    
    end
    
    maxvec = -(Teff)*loglambda;
    tracevec = zeros(K,1);
    for i = 1:K
        tmp = loglambda(i:K);
        tracevec(i,1) = -(Teff)*sum(tmp);
    end
    
    % Selection of column corresponding to ALPHA
    if alpha_crit == 0.10
        col = 1;
    elseif alpha_crit == 0.05
        col = 2;
    elseif alpha_crit == 0.025
        col = 3;
    elseif alpha_crit == 0.01
        col = 4;
    elseif alpha_crit == 0.001
        col = 5;
    end 
    %
      % Table really corresponding to model without constant
    % load % MAT file contains simulated critical values. 
    % load('CV_Johfull');
    % load('CV_Joh_200');
    load('CV_Johinfinite.mat');
    % load('CV_Johfull.mat');

    critmat = flipud(CV_Joh.tracecv);
    dim_critmat = size(critmat,1); 
    % Testing sequence is taken from Dietmar Bauer, many many thanks :)
    % TRACE TEST
    i = 1;
    while i <= K
        if i == 1
            if tracevec(i,1) <= critmat(dim_critmat-K+i,col)
                b_eta = 0;
                i = K+1;
            end
        end
        if i <= K
        if tracevec(i,1) > critmat(dim_critmat-K+i,col)
            i = i + 1;
        else 
            b_eta = i-1;
            i = K+1;
        end    
        if i == K
            if tracevec(i,1) > critmat(dim_critmat-K+i,col)
                b_eta = K;
                i = K+1;
            end            
        end    
        end
    end
    % MAX TEST
    % Tables for Critical Values: 1-\alpha: 90 - 95 - 97.5 - 99%
%{
    ximat = [61.96 65.3 68.35 72.36 
             56.09 59.06 61.57 65.21
             50.65 53.69 56.55 59.78 
             44.99 47.99 50.78 53.90
             38.98 41.51 44.28 47.15 
             33.62 36.36 38.59 41.00
             27.62 30.04 32.51 35.17 
             21.58 23.80 26.14 28.82
             15.59 17.89 20.02 22.99 
             9.52 11.44 13.27 15.69
             2.86 3.84 4.93 6.51];
%}
   % load max_critical_20.mat % MAT file contains simulated critical values. 

    maxmat = flipud(CV_Joh.maxeigcv);
    dim_maxmat = size(maxmat,1);  
    % MAX TEST
    i = 1;
    while i <= K
        if i == 1
            if maxvec(i,1) <= maxmat(dim_maxmat-K+i,col)
                b_xi = 0;
                i = K+1;
            end
        end
        if i <= K
        if maxvec(i,1) > maxmat(dim_maxmat-K+i,col)
            i = i + 1;
        else 
            b_xi = i-1;
            i = K+1;
        end          
        if i == K
            if maxvec(i,1) > maxmat(dim_maxmat-K+i,col)
                b_xi = K;
                i = K+1;
            end            
        end    
        end
    end
    
%}
    %
    % catch, if called with r==0 or r==K. 
    if b_eta==0 
        % in this case rank is restricted to zero. 
        alphahat = zeros(K,0);
        betahat = zeros(K,0);
  %      alpha_o = eye(s);
  %      beta_o = eye(s);
  %      Va = [];
    elseif b_eta==K
        % in this case there are not restrictions on the rank -> unrestricted
        % estimate. 
        alphahat = eye(K); 
        betahat2 = (S01*inv(S11))';
     %   betahat = (R1t\R0t)';
        betahat = (R1t\R0t); % estimate of beta transpose in the equation 
        %{

        alpha_o = zeros(s,0);
        beta_o = zeros(s,0);
        Va = [];
        %}
    else
        beta_f = Vmson; %when matlab solution
        %beta_f_temp = V;
        beta_f'*S11*beta_f;    % check for normalization must be = I.
        alpha_f = S01*beta_f;
        
        betahat = beta_f(:,1:b_eta);
        %alphahat1 = ((Z1t*betahat)\Z0t)';
        alphahat = alpha_f(:,1:b_eta);
        %alphahat3 = S01*betahat;
        %Bu part benim yazdigim bitti
    
   %     betahat = betahatbase;
   %     alphahat = alphahatbase;
   end

vres_temp = Z0t - Z1t*betahat*alphahat'; 
Gamma = (Z2t\vres_temp)';
vres = vres_temp - Z2t*Gamma';
% Covariance Matrix
Sigma = vres'*vres/Teff;
ldSigma = log(det(Sigma));
% log likelihood
LL_est = -0.5*(Teff*ldSigma+Teff*K);

% Number of parameters to estimate
parNu = K*K*q + 2*K*b_eta;

%AIC: Akaike Information Criterion
ic_Aic = ldSigma + parNu*(2/T);

%AIC: Corrected Akaike Information Criterion
ic_Caic = ic_Aic + 2*q*(q+1)/(T-q-1);
%ic_Caic = ic_Aic + (2*parNu)*(parNu+1)/T-parNu-1;

%BIC: Schwarz - Bayesian Information Criterion 
ic_SBic = ldSigma + parNu*(log(T)/T);

%HQC : Hannan-Quinn Criterion
ic_HQic = ldSigma + parNu*(2*log(log(T))/T);
RREst_info = 0;
%b_eta,alphahat,betahat
Est_Results = struct('EstInfo',RREst_info,'CodeName','johansenCoinTestbase.m');
Est_Results.EigVectors = Vmson;     % Normalized Eigenvectors
Est_Results.EigValues = lambda;     % Eigenvalues
Est_Results.EigIndx = sortIdxDm;    % Index of Eigenvalues (High to low)
Est_Results.CRank = b_eta;
Est_Results.CRankMeig = b_xi;
Est_Results.AlphaHat = alphahat;
Est_Results.BetaHat = betahat;
Est_Results.SRCoef = Gamma;
Est_Results.Residuals = vres;
Est_Results.TraceVal = tracevec;
Est_Results.MaxEigVal = maxvec;
Est_Results.LoglikeValue = LL_est;
Est_Results.SigmaHat = Sigma;
Est_Results.LogDetSigmaHat = ldSigma;
Est_Results.AICValue = ic_Aic;
Est_Results.AICcValue = ic_Caic;
Est_Results.SBICValue = ic_SBic; 
Est_Results.HQCValue = ic_HQic;
Est_Results.cValTrace = critmat;
Est_Results.cValMeig = maxmat;
if plotsRank>0
    figtrace = figure;
    hold on;
    set(gca,'XTick',0:K-1);
    bar(0:K-1,critmat(dim_critmat-K+1:dim_critmat,col),'b');
    plot(0:K-1,tracevec(1:end),'r*');
    title(sprintf('VECM trace test: Estimated coint rank: %d',b_eta));

    figmax = figure;
    hold on;
    set(gca,'XTick',0:K-1);
    bar(0:K-1,maxmat(dim_maxmat-K+1:dim_maxmat,col),'b');
    plot(0:K-1,maxvec(1:end),'r*');
    title(sprintf('VECM Max. test: Estimated coint rank: %d',b_xi));
    figure;
    hist(vres,20);
    title('Residuals VECM')
%    pict2 = figure;
%    title('Log(Det(Sigma))')

end
if printresults>1
    fprintf("Print results\n");
    fprintf('===================================================================================\n');
    fprintf('===================================================================================\n');

end %johCointest