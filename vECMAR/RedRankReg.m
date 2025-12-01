function [RREst_Results,Est_Results_Table,Est_details_output] = RedRankReg(R0t,R1t,alpha_crit,Joh_j)
%
% RedRankReg   Function to estimate reduced rank regresion and performs 
%                   trace statistics for cointegration rank.
%
% Syntax:
%
%   out = RedRankReg(R0t,R1t,alpha_crit,Joh_j);
%
% ------------------------------------------------------------
% Description:
%
%   RedRankReg     fuction estimates reduced rank regression for
%           Matrix Autoregresive Time Series 
% -------------------------------------------------------------------------
% Input Arguments:
%
%   R0t - Dependent Variable
%
%   R1t - Indepenednet Variable
%   
%   alpha_crit - Significance levels for the trace tests. Values can be 
%                   between 0.10 and 0.01. The default value is 0.05.
%   
%   Joh_j - Johansen 5 cases for models. No deterministic variables 
%                   included in this version!
%
% -------------------------------------------------------------------------
% Output Arguments:
%
%    RREst_Results - Results for estimated MARj(p) model. It is a 
%           structure with the following fields that correspond directly 
%           to those in the estimated model:
%
%       RREst_Results = struct('EstInfo',RREst_info,'CodeName','RedRankRegression.m');
%       RREst_Results.CRank = b_eta;
%       RREst_Results.MeigRank = b_xi;
%       RREst_Results.AlphaHat = alphahat;
%       RREst_Results.BetaHat = betahat;
%       RREst_Results.Residuals = vres;
%       RREst_Results.TraceVal = tracevec;
%       RREst_Results.cValTrace = critmat;
%       RREst_Results.cValMeig = maxmat;
%
%   RREst_Results_Table - Table of output. prints Not defined yet.
%
%   RREst_Results_details_output - Details of output. prints Not defined yet.
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
%   [2] Johansen, S. Likelihood-Based Inference in Cointegrated Vector
%       Autoregressive Models. Oxford: Oxford University Press, 1995.
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
[T,K] = size(R0t);

RREst_info = struct('EstimationMethod','Reduced Rank Regression',...
    'CodeName','RedRankReg.m','SampleSizeT',T,...
    'Dimension',K,'JohansonSet',Joh_j,'CriticalValpha',alpha_crit);
RREst_info.EstimatedData = {R0t,R1t};
RREst_info.readme = 'Output: Est_Resuts,Est_Resuts_Table,Est_details_output.'; 
% predefined space

%
    S11 = (R1t'*R1t)/T;
    S00 = (R0t'*R0t)/T;
    S10 = (R1t'*R0t)/T;
    S01 = (R0t'*R1t)/T;
    
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

    % Estimate the eigenvectors and eigenvalues of C = A*B':
 %       
        [Vm,Dm] = eig(S10*(S00\S01),S11);
        [lambda,sortIdx] = sort(diag(Dm),'descend'); % Ordered eigenvalues
        Vm = Vm(:,sortIdx); % Corresponding eigenvectors
                
        % Normalize the eigenvectors so that V'*S11*V = I, as in [2]:
        
        Um = Vm'*S11*Vm; % V diagonalizes S11
     %   Vmson = bsxfun(@rdivide,Vm,sqrt(diag(Um))');
        Vmson = Vm./sqrt(diag(Um))';
 %
    % Test statistics vectors, version with %(T-k) not% T
    switch Joh_j %additional vars included at first position due to deterministics? 
        case {2,4}
            loglambda = log(abs(ones(K,1)-lambda(1:end-1,:)));
    
        otherwise
            loglambda = log(abs(ones(K,1)-lambda));   % ABS for rounding lambda ~ 1    
    end
    
    maxvec = -(T)*loglambda;
    tracevec = zeros(K,1);
    for i = 1:K
        tmp = loglambda(i:K);
        tracevec(i,1) = -(T)*sum(tmp);
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
    % Table really corresponding to model without constant
    % load % MAT file contains simulated critical values. 
    % load('CV_Johfull');
    % load('CV_Joh_200');
    load('CV_Johinfinite.mat');
    critmat = flipud(CV_Joh.tracecv);
    dim_critmat = size(critmat,1);    
%    
    % Testing sequence is taken from Dietmar, Very very thanks!!
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
    %
 
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
    
    %
    % catch, if called with r==0 or r==s. 
    if b_eta==0 
        % in this case rank is restricted to zero. 
        alphahat = zeros(K,0);
        betahat = zeros(K,0);
    %
    elseif b_eta==K
        % in this case there are not restrictions on the rank -> unrestricted
        % estimate. 
        alphahat = eye(K); 
     %   betahat = (S01*inv(S11))';
        betahat = (S01/(S11))';
     %   betahat = (R1t\R0t)';
        betahat = (R1t\R0t); % estimate of beta transpose in the equation 
    else
    beta_f_temp = Vmson; %when matlab solution
    %beta_f_temp = V;
    beta_f_temp'*S11*beta_f_temp;    % check for normalization must be = I.
    beta_f = beta_f_temp;
    alpha_f = S01*beta_f_temp;
    
    betahatbase = beta_f(:,1:b_eta);
    %alphahat1 = ((Z1t*betahat)\Z0t)';
    alphahatbase = alpha_f(:,1:b_eta);
    %alphahat3 = S01*betahat;
    %Bu part benim yazdigim bitti

    betahat = betahatbase;
    alphahat = alphahatbase;
% --------------------------------------------------------------% 
    %normalize beta to beta' = [I_r, beta'_(K-r)].
%{
    TempNorm = rref(betahatbase');
    betahat = TempNorm';

    alphahat = ((R1t*betahat)\R0t)';
%}
  %  aa = S01*betahat;
% --------------------------------------------------------------% 
% --------------------------------------------------------------%    
    %normalize beta to orthonormal block column. 
%{  
    [~,R]= qr(betahat);
    Tr = R(1:b_eta,1:b_eta)';
    alphahat = alphahat*Tr;
    betahat = betahat*inv(Tr)';
%}
% --------------------------------------------------------------% 
    % This is another normalization --- ENDS 
%{
    %normalize beta to beta' = [I_r, beta'_(K-r)].
    % --- 2. Normalize A^T to [I_r, B^T] using QR ---
    beta_T = betahat';                        % A^T is r x k
    [Qb2, ~] = qr(beta_T(:, 1:b_eta));       % QR on first r columns of A^T
    beta2 = Qb2' * beta_T;                  % Apply transformation to A^T
    betaI(1:b_eta, 1:b_eta) = eye(b_eta);           % Force I_r block for clarity
    beta22 = A2(:, b_eta+1:end)';            % Extract B (transpose of B^T)
    %normalize beta to beta' Ends.
%}
    end

vres = R0t - R1t*betahat*alphahat';   

RREst_Results = struct('EstInfo',RREst_info,'CodeName','RedRankRegression.m');
RREst_Results.CRank = b_eta;
RREst_Results.MeigRank = b_xi;
RREst_Results.AlphaHat = alphahat;
RREst_Results.BetaHat = betahat;
RREst_Results.Residuals = vres;
RREst_Results.TraceVal = tracevec;
RREst_Results.cValTrace = critmat;
RREst_Results.cValMeig = maxmat;
% RREst_Results.rnew = r_new;

Est_Results_Table=0;
Est_details_output=0;
end