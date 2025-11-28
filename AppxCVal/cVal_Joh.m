function CV_Joh = cVal_Joh(numDims,T,rep);
% cVal_Joh approximates asymptotic critical values for Johansen trace  
%           statistics for High Dimensional Time Series        
% Syntax:
%
%   CV_Joh = cVal_Joh(p,T,rep,jcase);
% ------------------------------------------------------------
% Description:
%
%   cVal_Joh calculates approximations to the asymptotic distributions of 
%           the Trace and Maximum Eigenvalue test statistics
%       
% ------------------------------------------------------------
% Input Arguments:
%
%   numDims - Dimensions, default:2 ( Number of variables) 
%   T - number of observations, default:5000 ( Time dimension)
%   rep - number of replications, default:10000
% ------------------------------------------------------------
% Output Arguments:
%
% CV_Joh - Includes the Trace and Maximum eigenvalue critical values
%        - tracecv, rows: (p-r), columns [90 95 97.5 99 99.9 100] percentiles
%        - maxeigecv, rows: (p-r), columns [90 95 97.5 99 99.9 100] percentiles
%        - info, includes simulation information
% ------------------------------------------------------------
% Example:
% 
%    CV_Joh = cVal_Joh(20);
%    CV_Joh = cVal_Joh(20,5000,10000); 
% ------------------------------------------------------------
% Notes:
%
% These are the values from Johansen's 1995 book [1] and the MacKinnon 
%       values [2] for comparison to our approximations.
%
% Joh_cv = [ 2.98   4.14   7.02
%            10.35  12.21  16.16
%            21.58  24.08  29.19
%            36.58  39.71  46.00
%            55.54  59.24  66.71
%            78.30  86.36  91.12
%           104.93 109.93 119.58
%           135.16 140.74 151.70
%           169.30 175.47 187.82
%           207.21 214.07 226.95
%           248.77 256.23 270.47
%           293.83 301.95 318.14];
%
% MacKinnon_cv = [ 2.9762   4.1296   6.9406
%                10.4741  12.3212  16.3640
%                21.7781  24.2761  29.5147
%                37.0339  40.1749  46.5716
%                56.2839  60.0627  67.6367
%                79.5329  83.9383  92.7136
%                106.7351 111.7797 121.7375
%                137.9954 143.6691 154.7977
%                173.2292 179.5199 191.8122
%                212.4721 219.4051 232.8291
%                255.6732 263.2603 277.9962
%                302.9054 311.1288 326.9716];
% ------------------------------------------------------------
% References:
% 
%   [1] Johansen, S. Likelihood-Based Inference in Cointegrated Vector
%       Autoregressive Models. Oxford: Oxford University Press, 1995.
%
%   [2] MacKinnon, J. G., A. A. Haug, and L. Michelis. "Numerical
%       Distribution Functions of Likelihood Ratio Tests for
%       Cointegration." Journal of Applied Econometrics. v. 14, 1999,
%       pp. 563-577.
%
% ------------------------------------------------------------
%
% written by:
% Kurtulus Kidik and Dietmar Bauer, 
% Dept. of Economics
% Universität Bielefeld
% Universitätsstraße 25
% D-33615 Bielefeld
% dietmar.bauer@uni-bielefeld.de
% kurtulus.kidik@uni-bielefeld.de
% ------------------------------------------------------------
%
if nargin<4
    jcase = 1; % Johansen's deterministics
end
if nargin<3
    rep = 10000;
end
if nargin<2
    T = 5000; 
end
if nargin<1
    numDims = 2;
end

%randnumgen = rng("shuffle"); % for reproducibility
randnumgen = rng(1163504094,"twister");

alpha = [90 95 97.5 99]; 
% Corresponds [10% 5% 2.5% 1%],s.t. 1-alpha = [90% 95% 99%]
Quantiles_trace = zeros(numDims,size(alpha,2));
Quantiles_maxeig = zeros(numDims,size(alpha,2));
% Allvalues = cell(numDims,2); % this saves all results; however, it makes  
                               % the file huge.                                  
for r = 1:numDims 
    lam = zeros(rep,2);
    eigenvals = zeros(numDims,rep);
    for mi=1:rep
        clear y
        % generate data 
        u = randn(T,numDims);
        y(:,1:r)=cumsum(u(:,1:r));
        for ji=(r+1):numDims
            y(:,ji)=filter([1,-0.5],1,u(:,ji));
        end
        % calculate R0 and R1
        R0 = diff(y);
        R1 = y(1:end-1,:);
    
        % calculate Moment Matrices, Sij
        S00= R0'*R0;
        S01 = R0'*R1;
        S11 = R1'*R1;
        % Solve Eigenvalue Problem
        [~,D] = eig(S01'*(S00\S01),S11);    % Estimate the eigenvalues
        lam_sort = sort(abs(diag(D)),"descend");   % Ordered eigenvalues
        % For Trace 
        lam(mi,1) = -T*sum(log(abs(1-lam_sort((end-(r-1)):end)))); % abs for rounding lambda ~ 1
        % For Maximum
        lam(mi,2) = -T*log(abs(1-lam_sort(end-(r-1))));                 
        eigenvals(:,mi)= lam_sort;

    end
    %Allvalues(r,:) ={lam,eigenvals};  % please add if needed.
    Quantiles_trace(r,:) = prctile(lam(:,1),alpha);
    Quantiles_maxeig(r,:) = prctile(lam(:,2),alpha);

end

Desc = ['Asymptotic critical values for Johansen trace and maximum test ' ...
    'for cointegration by Kurtulus KIDIK'];
CV_Joh.tracecv = Quantiles_trace;
CV_Joh.maxeigcv = Quantiles_maxeig;
% CV_Joh.allvalues = Allvalues;     % please add if needed.
CV_Joh.info = struct('Description',Desc);
CV_Joh.info.CodeName = 'cVal_Joh.m';
CV_Joh.info.RandNumberInfo = randnumgen;
CV_Joh.info.Dimensions = numDims;
CV_Joh.info.NofObservation = T;
CV_Joh.info.NofRep = rep;
CV_Joh.info.JohansenCase = jcase;
end