function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);

% Initialize J as a vector before summing over all training examples
J = zeros(m,1);         

% Initialize matrix for binary outcome vector of all training examples
out = zeros(num_labels,m);    

DELTA1 = zeros(size(Theta1));
DELTA2 = zeros(size(Theta2));

Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% Do feedforward propagation for all training examples (m examples)
a1 = [ones(m,1) X];                 % a1: size m x (n+1)

z2 = Theta1 * a1';
a2 = [ones(m,1) sigmoid(z2)'];      % a2: size m x (size(Theta1,1)+1)

z3 = Theta2 * a2';
a3 = sigmoid(z3)';                  % a3: size m x num_labels (or size(Theta2,1))


for i=1:m                           % Go through all training examples 

    out(y(i),i) = 1;
    
    % First, sum over all output nodes (vectorized)
    J(i) = - 1/m * ( log(a3(i,:)) * out(:,i) + log(1 - a3(i,:)) * (1-out(:,i)) );
    
    delta3 =  a3(i,:)' - out(:,i);
    
    delta2 = Theta2' * delta3 .* a2(i,:)' .* (1 - a2(i,:)');
    delta2(1) = [];         % because a2(i,1) = 1, always, since it's a bias node.
    
    DELTA2 = DELTA2 + delta3 * a2(i,:);
    DELTA1 = DELTA1 + delta2 * a1(i,:);

end

% Add regularization term: square all Theta matrix entries,
% except those corresponding to the bias nodes (first column in Theta).
reg = lambda/(2*m) * ( sum(sum(Theta1(:,2:end).^2)) + sum(sum(Theta2(:,2:end).^2)) );

% Finally, compute cost function, J.
J = sum(J) + reg;


% For regularization, convert first column of Theta to 0s (bias nodes are not regularized).
temp1 = Theta1;                         temp2 = Theta2;
temp1(:,1) = 0;                         temp2(:,1) = 0;

Theta1_grad = 1/m .* DELTA1 + lambda/m * temp1; 
Theta2_grad = 1/m .* DELTA2 + lambda/m * temp2;

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];

end
