function [J, grad] = linearRegCostFunction(X, y, theta, lambda)
%LINEARREGCOSTFUNCTION Compute cost and gradient for regularized linear 
%regression with multiple variables
%   [J, grad] = LINEARREGCOSTFUNCTION(X, y, theta, lambda) computes the 
%   cost of using theta as the parameter for linear regression to fit the 
%   data points in X and y. Returns the cost in J and the gradient in grad

m = length(y); % number of training examples

J = 0;
grad = zeros(size(theta));

% General implementation of cost (appropriate for multivariate feature analysis)
J = 1/(2*m) * (X * theta - y)' * (X * theta - y) + ...
    lambda/(2*m) * theta(2:end)' * theta(2:end);

% We don't want to regularize theta(0)    (ie, theta(1) in Matlab indexing)
theta_temp = theta;
theta_temp(1) = 0;

% Vectorized gradient calculation
grad = 1/m * X' * ( X * theta - y ) + lambda/m * theta_temp;

grad = grad(:);

end
