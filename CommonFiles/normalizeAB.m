function [An,Bn] = normalizeAB(A,B)
%
% norm_MaTS provides the normalization of the autoregressive MaTS system. 
%
% Syntax:
%   [A,B] = norm_MaTS_syst(A,B);
%
% -------------------------------------------------------------------------
% Input Arguments:
%
%   A - three dimensional array of dimension M x N x j: MxN is the 
%           dimension of the matrix valued time series, with one lag and j
%           the maximal number of terms included. 
%   B - three dimensional array of dimension M x N x j.
% -------------------------------------------------------------------------
% Output Arguments:
%   OUTPUTS: normalized pair (A,B), for one lag.
%
% -------------------------------------------------------------------------
% REMARKS:
% It uses the rewriting into a matrix with the same entries as kron(B,A),
% but where each component adds a rank one entry. 
% Normalisation then is achieved using the QR decomposition. 
%
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
% 12.8.2024..
% -------------------------------------------------------------------------
%
dims = size(A);

M1 = size(A,1); %M2
M2 = size(A,2); %N2
N1 = size(B,1); % M1
N2 = size(B,2); % N1

if length(dims)>2
    J = dims(3);
else
    J = 1;
end 

% normalisation is performed lag by lag
KP = zeros(M1*N1,M2*N2);
Phi = zeros(N1*N2,M1*M2); 
        
for ji=1:J
    KronProd = kron(B(:,:,ji),A(:,:,ji));
        % convert to reordered matrix:
    KP = KP + KronProd;
end

% -- start
% Reshaping
rA = reshape(KP, [M1,N1,M2,N2]);
pA = permute(rA, [2 4 1 3]);
rpA = reshape(pA, N1*N2,M1*M2);
Phi = rpA;
% -- end
    [Q,R]= qr(Phi'); % we need transpose for QR

    % adjust sign of diagonal entries
    signsf = sign(diag(R(1:J,1:J)));
    signs = signsf;                 % 
    signs(signs == 0) = 1;          % 
    Qsign(:,1:J)=Q(:,1:J)*diag(signs);
    Rsign(1:J,:)=diag(signs)*R(1:J,:);
    % write result into matrices
    An = reshape(Qsign, M1, M2, J);
    Bn = reshape(Rsign', N1, N2, J);

end
