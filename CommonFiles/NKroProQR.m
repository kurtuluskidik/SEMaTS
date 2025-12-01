%% Finding Nearest Kronecker product of a matrix with QR decomposition 
function [D, C, Diff,err] = NKroProQR(A, SizeD, SizeC, r)
% SYNTAX: [D, C, RA, Q, R, Diff] = NKroProQR(A, [M1 N1], [M2 N2], r)
%
% Given A (M-by-N matrix) finds its Kronecker product of matrices 
%           D (M1-by-N1 matrix) and C (M2-by-N2 matrix).
%       e.i. A = kron(D,C)
%
% INPUT:       A ... M-by-N matrix.
%          SizeD ... [M1 N1].
%          SizeC ... [M2 N2].
%              r ... number of terms.
%
% OUTPUT:  D ... matrix M1-by-N1. 
%          C ... matrix M2-by-N2.
%
% REFERENCE: https://doi.org/10.1007/978-94-015-8196-7_17
%            Van Loan, C.F., Pitsianis, N. (1993). 
%           Approximation with Kronecker Products. In: Moonen, M.S., 
%           Golub, G.H., De Moor, B.L.R. (eds) Linear Algebra for Large 
%           Scale and Real-Time Applications. NATO ASI Series, vol 232. 
%           Springer, Dordrecht. https://doi.org/10.1007/978-94-015-8196-7_17
% 
%-----------------------------------------------------------
% AUTHOR: Kurtulus Kidik, 26.04.2024. 
%-----------------------------------------------------------

if nargin < 4
  r = 1;
end

if nargin < 3
  error('NKroPro error: at least 3 arguments required')
end

% Get sizes.
[M, N] = size(A);
M1 = SizeD(1);
N1 = SizeD(2);
M2 = SizeC(1);
N2 = SizeC(2);

% Make sure dimensions match.
if (M ~= M1*M2)
    error("Check dimensions M1 and M2!")
end
if (N ~= N1*N2)
    error("Check dimensions N1 and N2!")
end

% Reshaping
rA = reshape(A, [M2,M1,N2,N1]);
pA = permute(rA, [2 4 1 3]);
rpA = reshape(pA, M1*N1,M2*N2);
RA = rpA';

% using QR decompoistion
[Q,R]= qr(RA);

% adjust sign of diagonal entries
signsf = sign(diag(R(1:r,1:r)));
signs = signsf;
signs(signs == 0) = 1;
Qa(:,1:r)=Q(:,1:r)*diag(signs);
Ra(1:r,:)=diag(signs)*R(1:r,:);

% write result into matrices
D = reshape(Ra', M1, N1, r);
C = reshape(Qa, M2, N2, r);

    if nargout > 2
        KP = zeros(M1*M2,N1*N2);
        for k=1:r
            KronProd = kron(squeeze(D(:,:,k)),squeeze(C(:,:,k)));
        % convert to reordered matrix:
            KP = KP + KronProd;
        end
        Diff = A - KP; % difference error of matrices.
        %Diffqr = A - kron(Dqr, Cqr);
        err = norm(A - KP,"fro");
    end
end