function [p_vals, p] = predict(theta, X)
%PREDICT Predict whether the label is 0 or 1 using learned logistic 
%regression parameters theta
%   [p_vals p] = PREDICT(theta, X) computes the predictions for X using a 
%   threshold at 0.5 (i.e., if sigmoid(theta'*x) >= 0.5, predict 1)
%   p_vals: actual probability values obtained from logistic hypothesis function
%   p: converted p_vals to 0/1 based on threshold of 0.5

m = size(X, 1);                 % Number of training examples

p_vals = zeros(m, 1);

p = zeros(m, 1);

p_vals = sigmoid(X * theta);

p = p_vals;

p1 = find(p_vals >= 0.5);

p(p1) = 1;
p = floor(p);

end
