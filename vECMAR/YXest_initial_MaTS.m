%% Rank-r approximation with QR decomposition
% function [A,B] = initial_MaTS(Y,L,p)
%
% Function to estimates the initial values for MaTS with VAR 
%
% Input:   Y ... T x M x N array of observations.
%          L ... integer; lag length.
%          p ... integer; number of terms per lag.
% Output:   
%           A ... four dimensional array M x M x L x p. 
%           B ... four dimensional array N x N x L x p.
% External Functions: var, NKroProQR
%-----------------------------------------------------------
% AUTHOR: Kurtulus Kidik, 21.06.2024. 
%-----------------------------------------------------------
function [A, B, SigmaR, SigmaC] = YXest_initial_MaTS(Y,Z,L,p)

if nargin<3
    p = 1;    
end
if nargin<2
    L = 1;    
end
[Ty,My,Ny] = size(Y);
Ky = My*Ny;
[Tz,Mz,Nz] = size(Z);
Kz = Mz*Nz;
if Ty+L ~= Tz
  error('Time dimension does NOT match')
end
if Ny <= 1
  error('Not three dimensional array')
end
Teff = Ty;
y = reshape(Y,Ty,Ky);
z = reshape(Z,Tz,Kz);
%C = var(y,)

%Generate lagged regressor matrix
vZ = zeros(Tz,Kz*L);

for i = 1:L
    vZ(:,(1+(i-1)*Kz):i*Kz) = [NaN(i,Kz );z(1:end-i,:)];    
end
vZ = vZ((L+1):end,:);
Phi0 = (vZ\y)';
Phi = reshape(Phi0,Ky,Ky,L);
%Phi0 = (Z2t\Z0t);
% Compute Residulas  and SSE
vet = y - vZ * Phi0';
SSE = vet' * vet;
SigmaHat = SSE / Teff;

[SigmaC, SigmaR] = NKroProQR( SigmaHat, [Ny Ny], [My My], 1);
%R0t = Z0t - Z2t*(Z2t\Z0t); % Z0t on Z2t residuals
% generate R_{0t}
%R0t = Z0t - Z2t*Phi0;

% Covariance Matrix

%[A,B] = initial_MaTS(C,M,N,p);

[Ms,Ns, L] = size(Phi);
%[Ms,Ns, L] = size(C);

% Make sure dimensions match.
if (Ms ~= My*Ny)
    error("Check dimensions for A!")
end
if (Ns ~= Ny*My)
    error("Check dimensions for B!")
end
A = zeros(My,My,L,p);
B = zeros(Ny,Ny,L,p);
for j=1:L
    [B1, A1] = NKroProQR(squeeze(Phi(:,:,j)), [Ny Ny], [My My], p);
    for k=1:p      
        A(:,:,j,k) = squeeze(A1(:,:,k));
        B(:,:,j,k) = squeeze(B1(:,:,k));
    end
end

end
