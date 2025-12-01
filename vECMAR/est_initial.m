%% Initial Esimation 
function [A,B,All_Coeff] = est_initial(Y,L,p)
%[A,B,SigmaR,SigmaC] = est_initial_mecm(Y,L,p)

if nargin<3
    p = 1;    
end
if nargin<2
    L = 1;    
end
[T,M,N] = size(Y);
k = L-1;
if N <= 1
  error('Not three dimensional array')
end


%[~,Omega,~,C] = var_ols_est(y,L);
%
% ========VAR initials
% This is using VAR then coverts VECM
y = reshape(Y,T,M*N);
[VAROlsEst_Results,~] = var_ols_est(y,L);
%[A,B] = initial_MaTS(C,M,N,p);
Omega = VAROlsEst_Results.SigmaHat;
C = VAROlsEst_Results.VARCoefficients;
% =========VAR initials ENDS
%{
% ++++++++++++++Columnwise VAR Block Diagonal 
for ik = 1:N
    [VAROlsEst_Results,~] = var_ols_est(Y(:,:,ik),L);
    Ctemp(:,:,:,ik) = VAROlsEst_Results.VARCoefficients;
end
%perCtemp = permute(Ctemp,[1,2,4,3]);
%for ik = 1:N
C = zeros(M*N,M*N,L);
    for iik = 1:L
        outdiag = [];
        for ik = 1:N
            outdiag = blkdiag(outdiag,Ctemp(:,:,iik,ik));
        end
    
       C(:,:,iik) = outdiag;
    end
% ++++++++++++++Columnwise VAR Block Diagonal ENDS
%}
[Ms,Ns,L] = size(C);

% Make sure dimensions match.
if (Ms ~= M*N)
    error("Check dimensions for A!")
end
if (Ns ~= N*M)
    error("Check dimensions for B!")
end

% VAR to VECM
Pi = -eye(M*N) + sum(C,3);
All_Coeff(:,:,1) = Pi;

for ii = 1 : k
Gammas = zeros(Ms,Ns);
    for ij = ii : k
         Gammas = Gammas - C(:,:,(ij+1));         
    end
    All_Coeff(:,:,ii+1) = Gammas;
end
%{
for ii = 1 : k
    All_Coeff(:,:,ii+1) = Gammas(ii);
end
%}
% This is using VAR then coverts VECM +++ Ends +++
%
% This part uses full rank VECM estimation
%{
[~,alphahat,betahat,~,Phi,~,~,~] = VECM_I1(y,k,M*N,5);
mPhi = reshape(Phi, Ms,Ns,k);
All_Coeff1(:,:,1) = alphahat*betahat';
for ii = 1 : k
    All_Coeff1(:,:,ii+1) = mPhi(:,:,ii);
end
%}
% This part uses full rank VECM estimation +++ Ends +++
A = zeros(M,M,L,p);
B = zeros(N,N,L,p);
for ii=1:L

    [B1, A1] = NKroProQR(squeeze(All_Coeff(:,:,ii)), [N N], [M M], p);
    for k=1:p      
        A(:,:,ii,k) = squeeze(A1(:,:,k));
        B(:,:,ii,k) = squeeze(B1(:,:,k));
    end
%
end
%{
% If needed; Covariance Matrix is Kronecker product. NKP approximation of SigmaR and SigmaC 
Reordered_Omega = reorder_KronProd(Omega,N,M,N,M);
[Qs,Rs]= qr(Reordered_Omega');
signsigma = sign(diag(R(1,1)));
Qsigma = Qs(:,1)*diag(signsigma);
Rsigma = diag(signsigma)*Rs(1,:);
SigmaR = reshape(Qsigma,M,M);
SigmaC = reshape(Rsigma,N,N)';
%}
end
