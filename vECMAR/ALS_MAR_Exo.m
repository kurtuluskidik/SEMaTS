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

function [ExoMaTSEst_Results,ExoEst_Results_Table,ExoEst_details_output] = ALS_MAR_Exo(Y,X,jlist,Ainit,Binit,plots,printlevel,tol)
%
if nargin<8
    tol = 1.0e-06;%1.0e-10; % 0.000000001;%1e-6; %0.000000001; %1e-6; %0.000000001;
end
if nargin<7
    printlevel = 0;
end
if nargin<6
    plots = 0;
end
if nargin<3
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
[Ty,My,Ny] = size(Y);
[T,M,N] = size(X);
p = size(jlist,2);
J = max(jlist);
jsum = sum(jlist);
K = M*N;
Ky = My*Ny;
Teff = T-p;


if nargin<4
    [Ainibase,Binibase] = YXest_initial_MaTS(Y,X,p,J);
else
    Ainibase = Ainit;
    Binibase = Binit;
end

ranugeMultiMaTSest = rng;
MaTSEst_info = struct('EstimationMethod','Alternatimg Least Squares','CodeName','ALS_MAR_Exo.m','SampleSizeT',T,...
    'RowDimension',M,'ColumnDimension',N,'NumberOfLags',p,'NumberOfTerms',jlist,...
    'RandomNuGen',ranugeMultiMaTSest);
MaTSEst_info.InitialA = Ainibase;
MaTSEst_info.InitialB = Binibase;
%MaTSEst_info.EstimatedDataY = Y;
%MaTSEst_info.EstimatedDataX = X;
MaTSEst_info.readme = 'Output: Est_Resuts,Est_Resuts_Table,Est_details_output.'; 

ABini = cell(p,2);
%Bini = Binibase;
for li = 1:p
    ABini(li,1) = {squeeze(Ainibase(:,:,li,1:jlist(li)))};
    ABini(li,2) = {squeeze(Binibase(:,:,li,1:jlist(li)))};
end

Aest = [];
Best = [];
for li = 1:p
    Aest = [Aest reshape(ABini{li,1},M,M*jlist(li))];
    Best = [Best reshape(ABini{li,2},N,N*jlist(li))];
end

yv = reshape(Y,Ty,Ky);
Yt = permute(Y,[2,3,1]);
%Yt = mY(:,:,(p+1:end)); % effective sample size
mX = permute(X,[2,3,1]);

%Yt = mX(:,:,(p+1:end)); % effective sample size

Yt_2 = zeros(M*p,N*p,(Teff));
for ji = 1:p
    Yt_2((M*(ji-1))+1:M*ji,(N*(ji-1))+1:N*ji,:) = mX(:,:,(1+(p-ji)):(end-ji));
end

if J>1
    Ytemp= [];
    for ji = 1:p
        Ytemp = [Ytemp repmat(Yt_2((M*(ji-1))+1:M*ji,(N*(ji-1))+1:N*ji,:),1,jlist(ji))]; 
    end
    Ytcell = mat2cell(Ytemp, M, ones(1,sum(jlist))* N, Teff);
    Yt_1 = zeros(M*jsum,N*jsum,Teff);
    for ji = 1:jsum
        Yt_1((M*(ji-1))+1:M*ji,(N*(ji-1))+1:N*ji,:) = Ytcell{ji};
    end
end

if J<=1
    Yt_1 = Yt_2;
end
%CY = diag(Y(:,:,))
%sigmaC = eye(N);
%sigmaR = eye(M);
% iteration
err0 = norm(Y,"fro");

if printlevel>0
    FunctionValue = err0^2/(T*M*N);
    fprintf('===================================================================================\n');
    fprintf("   Iteration        f(x)          Improvement     Syst Diff.      Syst StepS.     Sigma     Sigma StepS.\n");
    fprintf("      %d             %f \n",0,FunctionValue);
end
err = err0;

it = 0;
change = 1;
EstPairs = ABini;
ldSigmahat0 = log(det((yv'*yv)/T));
ldSigmahat = ldSigmahat0;
diff_syst =T*K;
%
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

    % estimate B(1,1) for given A 
    
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

    changerr = err_old - err;
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
        fprintf("      %d             %f      %f        %f       %f     %f     %f\n",...
            it, FunctionValue, Improvment, diff_syst, StepSz_syst, ldSigmahat,StepSzSigma);
    end
    if ((Improvment < 1.0e-10) || (isinf(ldSigmahat)))

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
Aestarray = zeros(M,M,p,J);
Bestarray = zeros(N,N,p,J);
for ii = 1:p
    Aestarray(:,:,ii,1:jlist(ii)) = Aesttemp(:,:,1:jlist(ii));
    Aesttemp = Aesttemp(:,:,jlist(ii)+1:end);
    Bestarray(:,:,ii,1:jlist(ii)) = Besttemp(:,:,1:jlist(ii));
    Besttemp = Besttemp(:,:,jlist(ii)+1:end);
end

Phi = zeros(K,K,p); 
for li=1:p
    for jr=1:J
        Phi(:,:,li) = Phi(:,:,li) + kron(Bestarray(:,:,li,jr),Aestarray(:,:,li,jr));
    end
end
[~, ComMatrix] = VARCompMatrix(Phi);
EigenValCM = abs(eig(ComMatrix));
LambdaCM = max(EigenValCM);
StabilityCheck = lt(LambdaCM,1);
ExoMaTSEst_Results = struct('EstInfo',MaTSEst_info,'CodeName','ALS_MARdiffJSforMS_Mark3.m','LoglikelihoodValue',LL_est,...
    'SigmaHat',Sigmahat,'LogDetSigmaHat',ldSigmahat,'Ahat',Aestarray,'Bhat',Bestarray,'VARCoefficients',Phi,'Residuals',vres,...
    'MatsAIC',ic_Aic,'MatsAICc',ic_Caic,'MatsSBIC',ic_SBic,'MatsHQC',ic_HQic,'SampleSizeT',T,'RowDimension',M,...
    'ColumnDimension',N,'NumberOfLags',p,'NumberOfTerms',jlist,'Df',parNu,'Stability',StabilityCheck);
ExoEst_Results_Table = 0;
ExoEst_details_output = struct('Iteration',it,'MinPossible',change,'BreakReason',abs(StepSzSigma));
if plots>0
    figure;
    hist(vres,20);
    title('Residuals multi term MAR_j(p)')

end
end