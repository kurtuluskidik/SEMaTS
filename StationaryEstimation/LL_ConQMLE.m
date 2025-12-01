%% Funtion to minimize
% Loglikelihood funtion for H-VECM.
% INPUTS: Y       ... T x M x N real matrix of observations.
%         theta   ... d vector of parameters
%         M       ... integer; 
%         N       ... integer
%         L       ... integer; number of lags
%         p       ... integer; number of terms per lag.
%
% OUTPUTS: crit   ... real, value of criterion function.
%          ve     ... M*N x T real matrix of vectorized residuals.
%         

function [crit,ve2] = LL_ConQMLE(Y,theta,L,p)
%
%   
[T,M,N] = size(Y); 
[A,B] = param2MaTS(theta,M,N,L,p);
% set up vectorized system 
% for each lag a M*N x M*N matrix. 
vPsi = zeros(M*N,M*N,L); 
for l=1:L
    for jr=1:p
        vPsi(:,:,l) = vPsi(:,:,l) + kron(B(:,:,l,jr),A(:,:,l,jr));
    end
end
% calculate residuals 
ve = zeros(T,M*N); 
% initial vallues for residuals set to NaN
ve(1:L,:) = NaN;
vY = reshape(Y,T,M*N);

ve2 = vY((L+1):T,:);

for l=1:L
    ve2 = ve2 - vY((L+1-l):(T-l),:)*squeeze(vPsi(:,:,l))';
end

Sigmahat = ve2'*ve2/(T-L);

% calculate criterion value 
crit = log(det(Sigmahat)); % ignore trace

end