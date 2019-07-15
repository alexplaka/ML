function [J, grad] = lrCostFunction(theta, X, y, lambda)
%LRCOSTFUNCTION Compute cost and gradient for logistic regression with 
%regularization
%   J = LRCOSTFUNCTION(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

m = length(y); % number of training examples

J = 0;
grad = zeros(size(theta));

%

J = - 1/m * ( y' * log(sigmoid(X * theta)) + (1-y') * log(1 - sigmoid(X * theta)) ) + ...
    lambda/(2*m) * theta(2:end)' * theta(2:end);
% Did not regularize parameter theta_0, i.e., theta(1) in the above theta vector. 


% grad(1) = 1/m * ( sigmoid(X * theta) - y )' * X(:,1);
% 
% for j=2:size(X,2)
%     grad(j) = 1/m * ( sigmoid(X * theta) - y )' * X(:,j) + lambda/m * theta(j);
% end

% Vectorized gradient calculation
grad = 1/m * X' * ( sigmoid(X * theta) - y ) + lambda/m * theta;
grad(1) = 1/m * ( sigmoid(X * theta) - y )' * X(:,1);


grad = grad(:);

end
