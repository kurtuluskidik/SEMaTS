function Results = initialECMAR(Y,plist,itst)
%
% initialECMAR 
%
% Syntax: 
% 
%   Results = initialECMAR(Y,plist,itst);
%
% -------------------------------------------------------------------------
% Description:
%
%   initialECMAR initial estimation for vECMAR model:
%
% -------------------------------------------------------------------------
% Input Arguments:
%
%   Y1 - Data representing observations of a multi-dimensional time series
%       Y(t), specified as a three dimensional array of dimension T x M x N. 
%       T: Time dimension, R: Row dimension, M: Column dimension.
%       The last observation is the most recent. 
%
%   plist - number of terms per lag, vector of integers, i.e.: [1 2 3]
%   	    Defines also lag length
%
%   itst - iteration defined by main function. Related to rank of Pi matrix.
%
% -------------------------------------------------------------------------
% Output Arguments:
%
%   Results - Estimation residuals.
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
%
[T,M,N] = size(Y);
K = M*N;
k = size(plist,2);
p = k+1;
Teff = T-p;
%plist = [1 1];
%plist = [min(M*M,N*N) min(M*M,N*N)];
%jlist = [(k+2) plist]; 
DeltaY_t  = diff(Y,1,1);
Z0t = DeltaY_t;
Z1t = Y((k+1):end-1,:,:);
Z2t = DeltaY_t((k+1):end,:,:); % Not exactly as definition
Z2vt = reshape(Z2t,Teff,K);    % Not exactly as definition, this effective delta part
Z1vt = reshape(Z1t,Teff,K);

vdy = reshape(DeltaY_t,(T-1),K);
vdy = vdy((k+1):end,:);
mDY = permute(DeltaY_t,[2,3,1]);    %mY: Matrix form of Y (M by N by T)  
MaTSEst_Results = ALSforEC(Y,plist); % ,2,1
Ahat = MaTSEst_Results.Ahat;
Bhat = MaTSEst_Results.Bhat;
%
shortT = zeros(Teff,K);
for ii = 1:k
    %LagsTerms = zeros(M,N,Teff);
    for ik = 1:plist(ii)
        temp = pagemtimes(Ahat(:,:,(ii+1),ik),'none',mDY(:,:,(1+(k-ii)):(end-ii)),'none');
        temp2 = pagemtimes(temp,'none',Bhat(:,:,(ii+1),ik),'transpose');
        LagsTerms = permute(temp2,[3,1,2]);
        shortT = shortT + reshape(LagsTerms,Teff,K);
    end
end
vX1 = vdy - shortT;
PiMatrix = zeros(K);
if length(size(Ahat))<4
    PiMatrix = kron(Bhat(:,:,1),Ahat(:,:,1));
else
    %
    for ik = 1:size(Ahat,4)
            PiMatrix = PiMatrix + kron(Bhat(:,:,1,ik),Ahat(:,:,1,ik));
    end
end
%
if itst < 1
    R0t = vX1;

elseif itst < M*N
    [U,S,V] = svd(PiMatrix);
    alpha = U(:,1:itst)*S(1:itst,1:itst);
    beta = V(:,1:itst);
    vX1 = vdy - Z1vt*beta*alpha';
    Xt = (reshape(vX1,Teff,M,N));
    ExoMaTSEst_Results = ALS_MAR_Exo(Xt,Z0t,plist);
    R0t = ExoMaTSEst_Results.Residuals + Z1vt*beta*alpha';
else
    R0t = MaTSEst_Results.Residuals;
end
Results = struct('Residuals',R0t);

end