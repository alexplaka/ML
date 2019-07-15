function [theta, J_history] = gradientDescent(X, y, theta, alpha, num_iters)
%GRADIENTDESCENT Performs gradient descent to learn theta
%   theta = GRADIENTDESCENT(X, y, theta, alpha, num_iters) updates theta by 
%   taking num_iters gradient steps with learning rate alpha

m = length(y); % number of training examples
J_history = zeros(num_iters, 1);

    for iter = 1:num_iters

    %     theta = theta - alpha / m * ((X * theta - y)' * X)';   

        % Alternate implementation (only 1 transpose)
        theta = theta - alpha / m  *  X' * (X * theta - y);   
        %       2x1                 (mx2)' *  (mx1)
        %       2x1                       2x1
        % ============================================================

        % Save the cost J in every iteration    
        J_history(iter) = computeCost(X, y, theta);

    end

    figure('Name','Gradient Descent');     
    plot(J_history);
    xlabel('iterations');       ylabel('J');
    
end
