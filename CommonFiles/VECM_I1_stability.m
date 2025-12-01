%% Function to check VECM system satisfies stability
function [stable,CompMatrix,Lambdas] = VECM_I1_stability(alpha, beta, Gamma,sp,ploteig);
% VECM_I1_stability    Checks the stability condition of VECM system
% Syntax:
%
%   [checkst,CompMatrix] = VECM_I1_stability(alpha, beta, Gamma);
% ------------------------------------------------------------
% Description:
%
%   VECM_I1_stability Checks the stability condition of VECM system 
%       
% ------------------------------------------------------------
% Input Arguments:
%
%   alpha - 
%   beta - 
%   Gamma - Short run coefficients, Differenced part
%   sp - 
%   ploteig -
%   
% ------------------------------------------------------------
% Output Arguments:
%
%   checkst - Includes the Trace and Maximum eigenvalue critical values
%   CompMatrix
%   Lambdas
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
%   [2] Johansen, S. Likelihood-Based Inference in Cointegrated Vector
%       Autoregressive Models. Oxford: Oxford University Press, 1995.
%
% ------------------------------------------------------------
% Date:         01.08.2025
% written by:   Kurtulus Kidik 
%               Dept. of Economics
%               Universität Bielefeld
%               Universitätsstraße 25
%               D-33615 Bielefeld
%               kurtulus.kidik@uni-bielefeld.de
% ------------------------------------------------------------
%
    if nargin<3
        Gamma = [];
    end
    if nargin<4
        sp = 0.99;
    end
    if nargin<5
        ploteig = 0;
    end
    
    [Na,ra] = size(alpha);
    [Nb,rb] = size(beta);
    dimsG= size(Gamma);
    [N1,N2,k1] = size(Gamma);
    
    if (ra > Na) || (ra < 0) || (ra ~= rb)
      error('!!! Attention !!! : Check Dimensions alpha, beta and ranks !!!!!!')
    end
    
    if (Na ~= N1) || (Nb < N2) || (Na ~= Nb)
      error('!!! Attention !!! : Check Dimensions alpha, beta and Gamma !!!!!!')
    end
    N = N1;
    if length(dimsG)>=3
        k = (dimsG(3));
    elseif dimsG(1) == 0
        k = 0;
    else
        k = 1;
    end
    
    if k1 ~= k
        error('Wrong lag')
    end
    % CompMatrix = zeros(ra + (k)*N,ra + (k)*N);
    CM11 = eye(ra) + beta' * alpha;
    if k == 0
        CompMatrix = CM11;
    elseif k == 1
        CompMatrix = [ CM11, beta' * Gamma; alpha, Gamma];
    else
        FirstRow = CM11;
        for ki = 1:k
            FirstRow = [FirstRow, beta' * Gamma(:,:,ki)];
        end
        SecondRow = alpha;
        for ki = 1:k
            SecondRow = [SecondRow,Gamma(:,:,ki)];
        end
        CompMatrix = [ FirstRow; SecondRow;...
            zeros(N*(k-1),ra) eye(N*(k-1)) zeros(N*(k-1),N)];
    end
    % Find Eigenvalues
    eigenvals = eig(CompMatrix);
    Lambdas = sort(abs(eigenvals), 'descend');

    % Check for eigenvalues are in the unit circle
    if all(Lambdas < sp)
        stable = true;
    else
        stable = false;
    end
    
    % Plot eigenvalues if desired
    if ploteig>0
        theta = 0:pi/50:2*pi;
        r = 1;
        x = r*cos(theta);
        y = r*sin(theta);
        pict1 = figure;
        plot(x, y);
        hold on;
        plot(eigenvals,"o");
        grid on;
        axis equal; % Ensures the circle looks circular
    %    axis square;
        xline(0);
        yline(0);
        xlabel("Re(z)");
        ylabel("Im(z)");
        title('Stability of VECM System')
    end
end % END VECM_I1_stability