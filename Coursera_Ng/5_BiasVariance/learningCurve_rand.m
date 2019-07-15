function [error_train, error_val] = ...
    learningCurve_rand(X, y, Xval, yval, lambda)
%LEARNINGCURVE Generates the train and cross validation set errors needed 
%to plot a learning curve, while picking a random subset of training
%and validation examples.
%   [error_train, error_val] = ...
%       LEARNINGCURVE(X, y, Xval, yval, lambda) returns the train and
%       cross validation set errors for a learning curve. In particular, 
%       it returns two vectors of the same length - error_train and 
%       error_val. Then, error_train(i) contains the training error for
%       i examples (and similarly for error_val(i)).

% Number of training examples
m = size(X, 1);

% Number of validation examples
v = size(Xval, 1);
v_s = ceil(0.8 * v);         % choose subset size of validation examples

error_train = zeros(m, 1);
error_val   = zeros(m, 1);


for i=1:m
    x = randperm(i);         % create random sequence of subset of training examples
    z_temp = randperm(v);    % create random sequence of all validation examples
    z = z_temp(1:v_s);       % create random sequence of subset of validation examples

    theta = trainLinearReg(X(x,:),y(x),lambda);
    error_train(i) = linearRegCostFunction(X(x,:),y(x),theta,0);
    error_val(i) = linearRegCostFunction(Xval(z,:),yval(z,:),theta,0);
end



end
