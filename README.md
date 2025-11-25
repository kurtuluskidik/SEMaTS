# SEMaTS
This repository contains MATLAB codes for the paper "Specification and Estimation of Matrix Time Series Models with Multiple Terms" by Kurtulus Kidik and Dietmar Bauer. Here you will find codes for proposed estimation metods, replication code for the simulations and most of the tables and figures in the main paper. 

Matrix time series $(MaTS)$ models allow to provide models for time series in the situation where for a number of regions the same set of variables for each region is observed. 
This leads to the observations at every time point being represented as a matrix. In the literature, matrix autoregressive models have been proposed that use one term per time lag. This imposes strong restrictions of generality that can be alleviated by adding terms. More terms on the other hand require identification restrictions. In this paper we propose such restrictions for the stationary and the integrated case. 
For co-integrated processes the cointegrating relations can be estimated super-consistently as usual. To determine cointegration relations, we propose a novel estimation method for models with higher order lags $(p>1)$ with the extension of Johansen's framework. With this notion, we investigate and introduce new critical values for trace statistics and extend them to the use of higher dimensions. Beside the estimation we also discuss specification of the integer parameters such as the number of time lags as well as the number of terms per time lag. We show that the familiar information criterion based model selection methods have properties as in the general multivariate case.

Reference: 
Kidik, K. and Bauer, D."Specification and Estimation of Matrix Time Series Models with Multiple Terms"
