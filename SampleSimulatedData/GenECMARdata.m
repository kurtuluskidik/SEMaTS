function [simY,info_Data] = GenECMARdata(T,M,N,CovSet,q,ra,Re,SP)
%
% GenECMARdata    Data generation function for Cointegrated 
%                       Matrix-Valued Time Series
% 
% Syntax:
%
%   [simY,info_Data] = GenECMARdata(T,M,N,CovSet,p,ra,Re,SP);
% ------------------------------------------------------------
% Description:
%
%   GenVECMdata     Function for generating data for Cointegrated 
%           Matrix Autoregresive Time Series. Generates K = M*N -dimensional 
%           system with given rank, r and lag (q>1).
%
% ------------------------------------------------------------
% Input Arguments:
%   
%   T - Number of observations. 
%   M - Row dimension.
%   N - Column dimension.
%   CovSet - Covariance structure.Default: 2
%   q - number of lags for Short run coefficients, Differenced part. Default:1
%   ra - Desired cointegration rank. Default: 5000
%   Re - NUmber of simulated series. Default: 100
%   SP - Max Bound of non unit eigenvalue for Stability. Default: 0.95
%
% ------------------------------------------------------------
% Output Arguments:
%
%   simY - Re by 3 struct: 
%          o First column: Data representing observations of a 
%               multi-dimensional time series y(t), specified as a 
%               numObs-by-numDims matrix. (vectorized form.).
%          o Second column: struct: Generated matrices.
%          o Third column:Simulated Errors.
%
%   info_Data - 
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
%   [1] Kidik, K. and Bauer, D. "Specification and Estimation of Matrix 
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

% Choice of random generators   
ranugeMultiMaTS = rng("shuffle");   % For reproducibility
warning('off','econ:adftest:InvalidStatistic')
if nargin<8
    SP = 0.95;      % Max Bound of non unit eigenvalue for Stability         
end
if nargin<7
    Re = 100;
end
if nargin<6
    ra = 1;
end
if nargin<5
    q = 1;
end
if nargin<4
    CovSet = 2;
end
% ------------------------------------------------------------    
    % Structure of the sum of Kronecker products on short run part
jlist = sort(1:q,'descend');    % Number of terms in delta part [2 1]
Jmax = max(jlist);              % Max. value of Kronecker terms per Gamma_i
jsum = sum(jlist);
% ------------------------------------------------------------
 
K = M*N;

if max(jlist) > min(M*M,N*N)
    error('!!! Attention !!! Number of Terms can NOT be bigger than minimum of square of dimensions... !!!!!!')
end
if (ra > K) || (ra < 0)
  error('!!! Attention !!! : Check Dimensions and ranks !!!!!!')
end
if (SP > 1) || (SP < 0.1)
    error('!!! Attention !!! : Stability Number Wrong !!!!!!')
end

% For numerical stability
epsilonCov = 0.01; 
epsilonA = 1.0e-4;
epsilonB = 1.0e-4;
epsilonPi = 1.0e-4;
%epsilonPi = 1.0e-6; % rank = K, full
adj = 1; % r=0 ->2, r=2,12 ->1.5 r=12, ->1 , %19=[0.1,1]  ;%0.05;%0.30; 0,1=[0.15,1],
adjPi = 1; % r=0 ->2,  %1 19 icin iyi  0.2;;%0.5;%0.30;
%snr_target = 0.8;
%-------------------------------------------------------------------------
% Start Generating Data
%-------------------------------------------------------------------------
   
% Pre-allocate for the generated data
simY = cell(Re,3);          

PW = waitbar(0,'DG..Please wait...');
for re=1:Re
    waitbar(re/Re,PW,sprintf('Generating Cointegrated MaTS Data, Processing %d/%d,',re,Re));
        %
        adfresult = false;
        while ~adfresult
        %-------------------------------------------------------------------------
        % Draw Coefficients 
        %-------------------------------------------------------------------------
        stable = false;
        chkI1 = false;
        while ~stable || ~chkI1 
        % First Long-run coefficients, Alpha and Beta
                % --- Generate α and β from Haar (uniform orthogonal) distribution ---
                if ra == 0
                    % Rank is restricted to zero. No Cointegration in I(1) series.
                    alpha = zeros(K,ra);
                    beta = zeros(K,ra);
                    Pi = alpha * beta';
                %    alpha_o = eye(s);
                %    beta_o = eye(s);
                elseif ra == K
                    % Rank is unrestricted. Series are I(0) so that No need to VECM.
                    alpha = eye(K);
                %    alpha_o = zeros(s,0);
                %    beta_o = zeros(s,0);
                    crdetPi = 0;
                    while (crdetPi < epsilonPi)
                        [Uo1,Ro1] = qr(randn(K)); 
                        signRo1 = diag(sign(diag(Ro1)));
                        U2 = Uo1 * signRo1;
                        DM = (abs(1.8*rand(K,1) - 0.90)*adjPi);
                        crdetPi = min(DM);
                        if crdetPi < epsilonPi
                            DM(DM<epsilonPi) = DM(DM<epsilonPi) + epsilonPi;
                            A = diag(DM) * U2';
                            crdetPi = min(svd(A));
                        end
                        %
                        A = diag(DM) * U2';
                    end

                    beta = A;
                    Pi = alpha * beta';
                else
                    crdetPi = 0;
                    while (crdetPi < epsilonPi)
                        [Q1,R1] = qr(randn(K,ra),"econ"); % Haar-distributed orthogonal matrix
                        [Q2,R2] = qr(randn(K,ra),"econ");
                        signR1 = diag(sign(diag(R1(1:ra,1:ra))));
                        signR2 = diag(sign(diag(R2(1:ra,1:ra))));
                        beta = (Q1(:,1:ra)*signR1);
                        if sum(sum(round((beta'*beta),5) ~= eye(ra)))~=0
                            error('!!! Attention !!! : beta NOT Orthonormal !!!!!!')
                        end
                        alpha1 = (Q2(:,1:ra)*signR2);
                        AlphaChK = alpha1'*alpha1;
                        if sum(sum(round(AlphaChK,5) ~= eye(ra)))~=0
                            error('!!! Attention !!! : alpha NOT Orthonormal !!!!!!')
                        end

                       DM = abs(1.8*rand(ra,1) - 0.90)*adjPi;
                       alpha = (alpha1-beta)*diag(DM);
                       Pi = alpha * beta';
                       SvPi = svd(Pi);
                       crdetPi = min(SvPi(1:ra));

                        if crdetPi < epsilonPi
                            DM(DM<epsilonPi) = DM(DM<epsilonPi) + epsilonPi;
                            alpha = (alpha1-beta)*diag(DM);
                            Pi = alpha * beta';
                            SvPi = svd(Pi);
                            crdetPi = min(SvPi(1:ra));
                        end
                       crdetPi3 = abs(min(DM));
                    end                    
                end
                crdetPi1 = abs(det(beta'*alpha));
                eigPi = sort(abs(eig(beta'*alpha)),'descend');
                eigPi2 = sort(abs(eig(alpha*beta')),'descend');
                % Long-run ENDS
                %
        % Second Short run coeffients, Gamma's for delta lags
                %
                % Gerate Gamma matrices
                Abase = zeros(M,M,jsum);
                Bbase = zeros(N,N,jsum);
                % Each Coefficeients are randomly drawn
                for ik = 1:jsum
                    % For right coefiicient matrix, A matrix
                    crdetA = 0;
                    while (crdetA < epsilonA)
                        [Uo1,Ro1] = qr(randn(M)); 
                        [Uo2,Ro2] = qr(randn(M));
                        signRo1 = diag(sign(diag(Ro1)));
                        signRo2 = diag(sign(diag(Ro2)));
                        U1 = Uo1 * signRo1;
                        U2 = Uo2 * signRo2;
                        DMA = (1.8*rand(M,1) - 0.90)*adj;
                        crdetA = min(DMA);
                        if crdetA < epsilonA
                            DMA(DMA<epsilonA) = DMA(DMA<epsilonA) + epsilonA;               
                        end
                        crdetA = min(DMA);
                        Atemp = U1*diag(DMA)*U2';
                        
                        crdetA1 = abs(det(Atemp));
                        eigA = sort(abs(eig(Atemp)),'descend');
                    end
                    Abase(:,:,ik) = Atemp;
                    % For left coefiicient matrix, B matrix
                    crdetB = 0;
                    while (crdetB < epsilonB)
                        [Uo1,Ro1] = qr(randn(N)); 
                        [Uo2,Ro2] = qr(randn(N));
                        signRo1 = diag(sign(diag(Ro1)));
                        signRo2 = diag(sign(diag(Ro2)));
                        Q1 = Uo1 * signRo1;
                        Q2 = Uo2 * signRo2;
                        DMB = (1.8*rand(N,1) - 0.90)*adj;
                        crdetB = min(DMB);
                        if crdetB < epsilonB
                            DMB(DMB<epsilonB) = DMB(DMB<epsilonB) + epsilonB;               
                        end
                        crdetB = min(DMB);
                        Btemp = Q1*diag(DMB)*Q2';
                        
                        crdetB1 = abs(det(Btemp));
                        eigB = sort(abs(eig(Btemp)),'descend');

                    end
                    Bbase(:,:,ik) = Btemp;
                end
                % Fill the matrices 
                GammaA = zeros(M,M,q,jlist(1)); 
                GammaB = zeros(N,N,q,jlist(1));
                for ii = 1:q
                    GammaA(:,:,ii,1:jlist(ii)) = Abase(:,:,1:jlist(ii)); 
                    GammaB(:,:,ii,1:jlist(ii)) = Bbase(:,:,1:jlist(ii));
                    Abase = Abase(:,:,(jlist(ii)+1):end);
                    Bbase = Bbase(:,:,(jlist(ii)+1):end);
                end
    
                % --- Convert to VECM form coefficients ---
                % Transorm to VAR coefficients s.t. as sums of Kronecker products, Gamma_i = kron(B_i,A_i) 
                GammaBA = zeros(K,K,q); 
                for lt=1:q
                    for jr=1:jlist(lt)
                        GammaBA(:,:,lt) = GammaBA(:,:,lt) + kron(GammaB(:,:,lt,jr),GammaA(:,:,lt,jr));
                    end
                end

                % Checking system satisfies stability condition
                    if ra == 0
                        [stable, ~, Lambdas] = VARstability(GammaBA, SP);
                        chkI1 = true;
                    elseif K == ra
                        chkI1 = true;
                            % --- Convert to VAR form ---
                        varCoef = VECMsys2VARsys(alpha, beta, GammaBA);
              %          [stable, ~, Lambdas] = VARstability(varCoef, SP);
                        [stable, ~, Lambdas] = VECM_I1_stability(alpha, beta, GammaBA, SP);
                    else
                        [stable, ~, Lambdas] = VECM_I1_stability(alpha, beta, GammaBA, SP);
    
                        % Checking system satisfies solution to I(1)
                        alpha_o = null(alpha');
                        beta_o = null(beta');
                        solexistM = alpha_o'*(eye(K) - sum(GammaBA,3))*beta_o;
                        DsolexistM = det(solexistM);
                            %rank(solexistM,1.0e-6);
                        RsolexistM = rank(solexistM);
                        ckinf = isfinite(norm(inv(solexistM)));
                        
                        if ((abs(DsolexistM) > 1.0e-4) && (RsolexistM == K-ra) && (1 == ckinf))
                            chkI1 = true;
                        end
                    end
                
        end % Coefficients ENDS
        %-------------------------------------------------------------------------
        % Last check for stability
        check_ComM = le(Lambdas,1.0000);
        if sum(check_ComM) ~= (K*q+ra)
            error('!!! Attention !!! : Check for Stability !!!!!!')
        end
        %-------------------------------------------------------------------------
        % Generate Covariance Matrix
        %-------------------------------------------------------------------------
        % Determination of Covariance matrix, 
        % Here sigma defined as sigma = Cov(vec(E_t)), E_t are iid normal matrix form
        %---------------
        switch CovSet
            case 1 % Setting I: equals Identity matrix  
                Sigma = eye(K);
            case 2 % Setting II: Generic Covariance Structure: equals Q*D*Q'
                detsigma = 0;
                while detsigma<epsilonCov % || dsigma>99
                    [Qs1,Rs1] = qr(randn(K));
                    signRs1 = diag(sign(diag(Rs1)));
                    Q = Qs1*signRs1;
                   % D = abs(diag(randn(1,K)));
                    eigvalsD = 0.5 + rand(K, 1);  % Positive eigenvalues > 0
                    D = diag(eigvalsD);
                    Sigma1 = Q * D * Q';
                    %Sigma = Sigma1*2;
                    Sigma = Sigma1;
                    detsigma = det(Sigma);
                end
            case 3 % Setting III: Kronecker product of Sigma_c and Sigma_r 
                dsigmaR = 0;
                dsigmaC = 0;
                detsigma = 0;
               % while dsigma<epsilonCov
                while dsigmaR<epsilonCov || dsigmaC<epsilonCov
                    % For Sigma_c
                    [Qc1,Rc1] = qr(randn(N));
                    signRc1 = diag(sign(diag(Rc1)));
                    QC = Qc1*signRc1;
                    DC = 1.5 + rand(N, 1);  % Positive eigenvalues > 0
                    SigmaC1 = QC * diag(DC) * QC';
                    
                    % For Sigma_r
                    [Qr2,Rr2] = qr(randn(M));
                    signRr2 = diag(sign(diag(Rr2)));
                    QR = Qr2(:,1:M)*signRr2;
                    DR = 0.5 + rand(M, 1);  % Positive eigenvalues > 0
                    SigmaR1 = QR * diag(DR) * QR';
    
                    SigmaC = SigmaC1;
                    SigmaR = SigmaR1/norm(SigmaR1,"fro"); % Normalization ||A||_F =1 
  
                    dsigmaR = det(SigmaR);
                    dsigmaC = det(SigmaC);
                    detsigma = (dsigmaC^M)*(dsigmaR^N);
   
                end
                Sigma = kron(SigmaC, SigmaR);
        end % Covariance ENDS
        %-------------------------------------------------------------------------
        % Generate data
        %-------------------------------------------------------------------------
        if CovSet==3 
            cSigmaR = chol(SigmaR,"lower");
            cSigmaC = chol(SigmaC,"lower");
         %   cSigmaR = chol(SigmaR);
          %  cSigmaC = chol(SigmaC);
        else
            cSigma = chol(Sigma);
        end
        % Initialize f
        Leff = q+1;
        y = zeros(T+Leff,K);
        dy = zeros(T+Leff,K);
        epsilontemp = zeros(T+Leff,K);
        % Initial conditions
        for ix = 1:q
            y(ix, :) = randn(1,K);  
        end
    
        % iterate over time 
        for t = Leff+1:T+Leff
            dy(t, :) = y(t-1, :) * Pi';
            for ji = 1:q
                dy(t, :) = dy(t, :) + (y(t-ji, :) - y(t-ji-1, :)) * GammaBA(:,:,ji)';
            end
            y(t, :) = y(t-1, :) + dy(t, :);
            % Draw error    
            if CovSet==3 
                Z = randn(M,N);
                Mnoise = cSigmaR * Z * cSigmaC';
                noise = reshape(Mnoise,1,K);
            else
                noise = randn(1,K) * cSigma;
            end
            %
            % Add noise
            y(t,:) = y(t,:) + noise;
            epsilontemp(t,:) = noise;
        end
        y = y(Leff+1:end,:);
        errepsilon = epsilontemp(Leff+1:end,:);
        %
        adf200 = zeros(K,1);
        adf500 = zeros(K,1);
        adffull = zeros(K,1);
        
        for ia = 1 : K
            %{
            adf200(ia,:) = adftest(y(1001:1200,ia), 'Lags',0:3);
            adf500(ia,:) = adftest(y(1001:1500,ia), 'Lags',0:3);
            adffull(ia,:) = adftest(y(1001:4500,ia), 'Lags',0:3);
            %}
            % For T = 200
            [h2, ~, ~, ~, reg2] = adftest(y(1001:1200,ia), 'lags',0:15);
            listBIC2 = cat(2, reg2.BIC);
            [valb2, indb2] = min(listBIC2);
            adf200(ia,:) = h2(indb2);
            % For T = 500
            [h5, ~, ~, ~, reg5] = adftest(y(1001:1500,ia), 'lags',0:15);
            listBIC5 = cat(2, reg5.BIC);
            [valb5, indb5] = min(listBIC5);
            adf500(ia,:) = h5(indb5);
            % For full, T = 3500
            [h, ~, ~, ~, reg] = adftest(y(1001:4500,ia), 'lags',0:15);
            listBIC = cat(2, reg.BIC);
            [valb, indb] = min(listBIC);
            adffull(ia,:) = h(indb);
        end
        dadf200 = zeros(K,1);
        dadf500 = zeros(K,1);
        dadffull = zeros(K,1);
        for ia = 1 : K
            %{
            adf200(ia,:) = adftest(y(1001:1200,ia), 'Lags',0:3);
            adf500(ia,:) = adftest(y(1001:1500,ia), 'Lags',0:3);
            adffull(ia,:) = adftest(y(1001:4500,ia), 'Lags',0:3);
            %}
            % For T = 200
            [h2, ~, ~, ~, reg2] = adftest(diff(y(1001:1200,ia)), 'lags',0:15);
            listBIC2 = cat(2, reg2.BIC);
            [valb2, indb2] = min(listBIC2);
            dadf200(ia,:) = h2(indb2);
            % For T = 500
            [h5, ~, ~, ~, reg5] = adftest(diff(y(1001:1500,ia)), 'lags',0:15);
            listBIC5 = cat(2, reg5.BIC);
            [valb5, indb5] = min(listBIC5);
            dadf500(ia,:) = h5(indb5);
            % For full, T = 3500
            [h, ~, ~, ~, reg] = adftest(diff(y(1001:4500,ia)), 'lags',0:15);
            listBIC = cat(2, reg.BIC);
            [valb, indb] = min(listBIC);
            dadffull(ia,:) = h(indb);
        end
        level = sum((adf200+adf500+adffull),'all') == 0;
        dlevel = sum((dadf200+dadf500+dadffull)) == 3 * K;
        if level == 1 && dlevel == 1
            adfresult = true;
        end
    %{
     for ia = 1 : K
            dadf200(ia,:) = adftest(diff(y(1001:1200,ia)), 'Lags',0:15);
            dadf500(ia,:) = adftest(diff(y(1001:1500,ia)), 'Lags',0:3);
            dadffull(ia,:) = adftest(diff(y(1001:4500,ia)), 'Lags',0:3);
        end
    %}
     %   if sum(adf200+adf500+adffull) == 0 && sum((zeros(K,4)+3)-dadf200-dadf500-dadffull) == 0
     %       adfresult = true;
     %   end

        end
%}
        simY(re,1) = {y};                                         
        if CovSet == 3
            simC = struct('alpha',alpha,'beta',beta,'VecCoef',GammaBA);
            simC.RowA = GammaA;
            simC.ColumnB = GammaB;            
            simC.Omega = Sigma; 
            simC.SigmaRow = SigmaR;
            simC.SigmaCol = SigmaC; % Store for data as { A B Gamma} cells
        else
            simC = struct('alpha',alpha,'beta',beta,'VecCoef',GammaBA,'Omega',Sigma);         % Store for data as { A B Gamma} cells
            simC.RowA = GammaA;
            simC.ColumnB = GammaB; 
        end
        simY(re,2) = {simC};
        simY(re,3) = {errepsilon};
 end
    
    info_Data = struct('CodeName','GenECMARdata.m');
    info_Data.NumofObs = T;
    info_Data.Dimension = K;
    info_Data.NumberOfLags = q;
    info_Data.CovSetting = CovSet;
    info_Data.RankofPi = ra;
    info_Data.NumberOfRepetition = Re;
    info_Data.SpectrialRadius = SP;
    info_Data.RandomNuGen = ranugeMultiMaTS;
    info_Data.AdjustemtPi = adjPi;
    info_Data.AdjustemtCoef = adj;
    info_Data.readme = 'In generated simY, First column includes TxK vectorised series, Second column includes related coefficients and simga.';
    
    warning('on','all')
    waitbar(re/Re,PW,sprintf('Generating Cointegrated MaTS Data, Completed %d/%d,',re,Re));
end % END GenECMAR_data 