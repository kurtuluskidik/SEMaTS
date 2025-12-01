%% Alternating Least Squares Estimation for Matrix Time Series
% Function to estimate multi-term MAR(Y) 

% Y_t = A_1*X_t*B_1'+ A_2*X_t*B_2'+ ... + E_t

% SYNTAX: [Aest,Best] = alt_LS_MaTS(Y,X,L,p);
%
% INPUTS:  Y      ... three dimensional array of dimension T x M x N.
%          X      ... three dimensional array of dimension T x M x N.
% INPUT:   Y           ... T x M x N array of observations.
%          L           ... integer; lag length.
%          p           ... integer; number of terms per lag.
%          Z           ... T x Mz x Nz array of observations of exogenous vars.
%          Lz           ... integer; lag length for exo part.
%          pz           ... integer; number of terms per lag for exo part.
%          tol         ... real; tolerance for subsequent iterations
%          maxit       ... integer; maximum number of iterations 
%          printlevel ... integer; indicating, what information should be
%                          provided in the iterations,0,1,2.
%
% OUTPUT:  Aest ... four dimensional array M x M x x p. 
%          Best ... four dimensional array N x N x x p.
% 
% NOTE: No deterministic variables included in this version!
%-----------------------------------------------------------
% AUTHOR: Kurtulus Kidik, 21.06.2024. 
%-----------------------------------------------------------
%
function [MECMEst_Results,Est_Results_Table,Est_details_output] = ALSforEC(Y1,plist,plots,printlevel,tol)
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
    plist = 1;
end  

if plots>1
    pict1 = figure;
    title('Log Likelihood Value')
    pict2 = figure;
    title('Log(Det(Sigma))')
    pict3 = figure;
    title('Estimated System Value')
end

[T,M,N] = size(Y1);
K = M*N;

yv = reshape(Y1,T,K);      %y: Vectorized Y (T by M*N)
mY = permute(Y1,[2,3,1]);  %mY: Matrix form of Y (M by N by T)      
k = size(plist,2);
%plist = [1 1];
%plist = [min(M*M,N*N) min(M*M,N*N)];
psum = sum(plist);% Lag order of ECMAR Delta part.
jlist = [(k+2) plist];     % Term number of ECMAR (EC part and Delta part).
%jlist = [(k) plist];  
%jlist = [min(M*M,N*N) plist];
%jlist = [1 plist];
p = size(jlist,2);
Jmax = max(jlist);            % Max term number of all of terms
jsum = sum(jlist);         % Sum of term numbers
if Jmax>min(M*M,N*N)
    sprintf('!!! Attention !!! Number of Terms can NOT be bigger than minimum of square of dimensions... !!!!!!')
    sprintf('Choose Max. term=: %d',min(M*M,N*N))
    jlist = [min(M*M,N*N) plist];
end
Teff = T-(k+1);            % Effective Sample size         

% Generate Differences, Lagged Level and Lagged Differences
y_1 = yv(1:end-1,:);                   % Vectorized Y_1 (T-1 by M*N)
dy = diff(yv,1,1);                     % Differenced y (T-1 by M*N)
DeltaY_t = diff(mY,1,3);               % DeltaY_t. (T-1 by M by N)
Z0t = DeltaY_t;
%Z0t = permute(DeltaY_t,[2,3,1]);       % Matrix form DeltaY_t (M by N by T-1)
Z1t = mY(:,:,(1:end-1));               % Matrix form Y_1 (M by N by T-1)
%dy_test = reshape(DeltaY_t,T-1,M*N);  %Deltay_t: Vectorized DeltaY_t (T-1 by M*N)
%----------------------------
%Initial Estimation
[Ainibase,Binibase] = est_initial(Y1,p,Jmax);

ranugeMultiMaTSest = rng;
mecmMaTSEst_info = struct('EstimationMethod','Alternatimg Least Squares','CodeName','ALS_mecm.m','SampleSizeT',T,...
    'RowDimension',M,'ColumnDimension',N,'NumberOfLags',k,'NumberOfTerms',jlist,...
    'RandomNuGen',ranugeMultiMaTSest);
mecmMaTSEst_info.InitialA = Ainibase;
mecmMaTSEst_info.InitialB = Binibase;
%mecmMaTSEst_info.EstimatedData = Y1;
mecmMaTSEst_info.readme = 'Output: Est_Resuts,Est_Resuts_Table,Est_details_output.';
%
ABini = cell(p,2);
%Bini = Binibase;
for li = 1:p
    ABini(li,1) = {squeeze(Ainibase(:,:,li,1:jlist(li)))};
    ABini(li,2) = {squeeze(Binibase(:,:,li,1:jlist(li)))};
end
%    
Aest = [];
Best = [];
for li = 1:p
    Aest = [Aest reshape(ABini{li,1},M,M*jlist(li))];
    Best = [Best reshape(ABini{li,2},N,N*jlist(li))];
end

Yt = Z0t(:,:,(k+1:end)); % DeltaYt with effective sample size
Z1t = Z1t(:,:,(k+1:end)); % Y_1 with effective sample size
%
% Form EC part
%
mZ1t = zeros(M*jlist(1),N*jlist(1),Teff);
for tt = 1:Teff
    mZ1t(:,:,tt) = kron(eye(jlist(1)),squeeze(Z1t(:,:,tt)));
end

if k<1
    Yt_1 = mZ1t;
else
    %
    Yt_2 = zeros(M*k,N*k,(Teff));
    for ji = 1:k
        Yt_2((M*(ji-1))+1:M*ji,(N*(ji-1))+1:N*ji,:) = Z0t(:,:,(1+(k-ji)):(end-ji));
    end
    
    if plist(1)>1

        Ytemp= [];
        for ji = 1:k
            Ytemp = [Ytemp repmat(Yt_2((M*(ji-1))+1:M*ji,(N*(ji-1))+1:N*ji,:),1,plist(ji))]; 
        end
        Ytcell = mat2cell(Ytemp, M, ones(1,sum(plist))* N, Teff);
        Yt_1temp = zeros(M*psum,N*psum,Teff);
        for ji = 1:psum
            Yt_1temp((M*(ji-1))+1:M*ji,(N*(ji-1))+1:N*ji,:) = Ytcell{ji};
        end
    else
        Yt_1temp = Yt_2;
    end
    for tt = 1:Teff
        Yt_1(:,:,tt) = blkdiag(squeeze(mZ1t(:,:,tt)),squeeze(Yt_1temp(:,:,tt)));
    end
end

% iteration
err0 = norm(Yt,"fro");

if printlevel>0
    FunctionValue = err0^2/(T*M*N);
    fprintf('===================================================================================\n');
    fprintf("   Iteration        f(x)          Improvement     Step Size\n");
    fprintf("      %d             %f \n",0,FunctionValue);
end
err = err0;

it = 0;
change = 1;
EstPairs = ABini;
ldSigmahat0 = log(det((dy'*dy)/(T-1)));
ldSigmahat = ldSigmahat0;
diff_syst =(T-1)*M*N;


while (change>tol)
        %&&(it<maxit))
    % update 
    Aold = Aest;
    Bold = Best;
    EstPairsOld = EstPairs;
    %sigmaRold = sigmaR;
    %sigmaCold = sigmaC;
    err_old = err;
    diff_syst_old = diff_syst;
    ldSigmahatOld = ldSigmahat;
    %LLest_old = LL_est;

    % estimate A(1,1) for given B

    %parta1 = pagemrdivide(Yt,sigmaC);
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
    estYhat = pagemtimes((pagemtimes(Aest,'none',Yt_1,'none')),'none',Best,'transpose');
    Rt = Yt - estYhat;
    %TRt = pagetranspose(Rt);
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
    diff_syst = aNORM + bNORM;
    StepSz_syst = (diff_syst_old - diff_syst)/diff_syst_old;

    err = norm(Rt,"fro");
    vres = reshape(permute(Rt,[3,1,2]),Teff,K);
    Sigmahat = vres'*vres/Teff;
    ldSigmahat = log(det(Sigmahat));
    StepSzSigma = (ldSigmahatOld - ldSigmahat)/ldSigmahatOld;
    changerr = err_old -err;

    change = max(changerr,diff_syst);

    if plots>1
 
        % calculate LogLikelihood
        LL_est = -0.5*(Teff*ldSigmahat+trace(vres/(Sigmahat)*vres'));
        %
        set(0, 'CurrentFigure', pict1)
        hold on
        grid on
        scatter(it,LL_est,'filled');
        
        set(0, 'CurrentFigure', pict2)
        hold on
        grid on
        plot(it,ldSigmahat,'b*');

        
        %
        %
        norm_estYhat = norm(estYhat,"fro");
        norm_Yt = norm(Yt,"fro");
        set(0, 'CurrentFigure', pict3)
        hold on
        grid on
        plot(it,norm_estYhat,'r*');
        scatter(it,norm_Yt,'filled');
        %plot(it,norm(Yt,"fro"),'b*');
 
    end
 
    Improvment = (err_old - err)^2/(err_old);
    %
    it = it+1;
    % print out results, if wanted 
    if printlevel>0
        FunctionValue = err^2/(T*K);
        %fprintf("   Iteration        f(x)          Improvement     Step Size\n");
        fprintf("      %d             %f      %f        %f   %f   %f   %f\n",...
            it, FunctionValue, Improvment, diff_syst, StepSz_syst, ldSigmahat,StepSzSigma);
    end

    if (Improvment < 1.0e-10) || (isinf(ldSigmahat))
        break
    end
    if ((abs(StepSz_syst) < 1.0e-6) || it>1000)
        break
    end

end
if printlevel>0
    fprintf('===================================================================================\n');
    fprintf('===================================================================================\n');
end
 
%
vres = reshape(permute(Rt,[3,1,2]),Teff,K);
Sigmahat = vres'*vres/Teff;
ldSigmahat = log(det(Sigmahat));

% calculate LogLikelihood
LL_est = -0.5*(Teff*ldSigmahat+trace(vres/(Sigmahat)*vres'));

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
ic_Caic = ic_Aic + 2*k*(k+1)/(T-k-1);
%ic_Caic = ic_Aic + (2*parNu)*(parNu+1)/T-parNu-1;

%BIC: Schwarz - Bayesian Information Criterion 
ic_SBic = ldSigmahat + parNu*(log(T)/T);
%ic_Caic = ic_SBic + parNu;

%HQC : Hannan-Quinn Criterion
ic_HQic = ldSigmahat + parNu*(2*log(log(T))/T);
% VAR coefficients such that, Phi = kron(B,A)
Aesttemp = reshape(Aest,M,M,jsum);
Besttemp = reshape(Best,N,N,jsum);
Aestarray = zeros(M,M,p,Jmax);
Bestarray = zeros(N,N,p,Jmax);
for ii = 1:p
    Aestarray(:,:,ii,1:jlist(ii)) = Aesttemp(:,:,1:jlist(ii));
    Aesttemp = Aesttemp(:,:,jlist(ii)+1:end);
    Bestarray(:,:,ii,1:jlist(ii)) = Besttemp(:,:,1:jlist(ii));
    Besttemp = Besttemp(:,:,jlist(ii)+1:end);
end
%Aestarray = reshape(Aest, M,M,L,p);
%Bestarray = reshape(Best, N,N,L,p); 

Phi = zeros(K,K,p); 
for li=1:p
    for jr=1:Jmax
        Phi(:,:,li) = Phi(:,:,li) + kron(Bestarray(:,:,li,jr),Aestarray(:,:,li,jr));
    end
end
[~, ComMatrix] = VARCompMatrix(Phi);
EigenValCM = abs(eig(ComMatrix));
LambdaCM = max(EigenValCM);
StabilityCheck = lt(LambdaCM,1);
MECMEst_Results = struct('EstInfo',mecmMaTSEst_info,'CodeName','ALS_MARdiffJSforMS_Mark3.m','LoglikelihoodValue',LL_est,...
    'SigmaHat',Sigmahat,'LogDetSigmaHat',ldSigmahat,'Ahat',Aestarray,'Bhat',Bestarray,'VARCoefficients',Phi,'Residuals',vres,...
    'MatsAIC',ic_Aic,'MatsAICc',ic_Caic,'MatsSBIC',ic_SBic,'MatsHQC',ic_HQic,'SampleSizeT',T,'RowDimension',M,...
    'ColumnDimension',N,'NumberOfLags',k,'NumberOfTerms',jlist,'Df',parNu,'Stability',StabilityCheck);
Est_Results_Table = 0;
Est_details_output = struct('Iteration',it,'MinPossible',change,'BreakReason',abs(StepSzSigma));
if plots>0
    figure;
    hist(vres,20);
    title('Residuals multi term MAR_j(p)')

end
end