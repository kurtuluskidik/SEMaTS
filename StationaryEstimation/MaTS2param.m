function theta = MaTS2param(A,B)
% MaTS2param calculates the parameters corresponding to the p terms in (A,B) for L lags.
%
% SYNTAX: [theta] = MaTS2param(A,B);
%
% INPUTS: A ... M x M x L x p real array of p terms.
%         B ... N x N x L x p real array of p terms.
%        
% OUTPUTS: theta ... d vector of parameters.
%
% REMARKS: used in MaTS Y_t = \sum_{l=1}^L \sum_{j=1}^p A_{l,j} Y_{t-l} B_{l,j}' + E_t.
% provides parameters (using the QR decomposition for each lag) 
% for p terms matrices.
% 
% AUTHOR: dbauer, 21.8.2023.

L = size(A,3);

theta = [];
for l=1:L
    theta = [theta(:);term_param(squeeze(A(:,:,l,:)),squeeze(B(:,:,l,:)))];
end
