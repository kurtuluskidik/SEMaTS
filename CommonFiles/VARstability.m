%% Function to check VAR system satisfies stability
function [stable, CompMatrix, Lambdas] = VARstability(VarCoef,sp,ploteig);
%
% VARstability   Checks the stability condition of VAR system
% Syntax:
%
%                               stable = VARstability(VarCoef);
%        [~, CompanionMatrix, lambdas] = VARstability(VarCoef,sp,ploteig);
%   [stable, CompanionMatrix, lambdas] = VARstability(VarCoef,sp,ploteig);
% ------------------------------------------------------------
% Description:
%
%   VARstability Check for system stability. 
%       
% ------------------------------------------------------------
% Input Arguments:
%
%   VarCoef - three dimensional coefficinet array.
%   sp      - Stability point
%   ploteig - Plots eigenvalues.
%   
% ------------------------------------------------------------
% Output Arguments:
%
%   stable - If stable returns 1
%   CompMatrix -
%   Lambdas -
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
    if nargin<2
        sp = 0.99;
    end
    if nargin<3
        ploteig = 0;
    end

    dims = size(VarCoef);   
    K1 = size(VarCoef,1);
    K2 = size(VarCoef,2);
    
    % Determine the number of lags
    if length(dims) >= 3
        L = (dims(3));
    else
        L = 1;
    end

    % Construct companion matrix
    if L > 1
        firstrow = reshape(VarCoef, K1, K2*L);
        CompMatrix = [ firstrow; eye(K1*(L-1)) zeros(K1*(L-1), K2)];
    else
        CompMatrix = VarCoef;
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



end