function [theta, J_history] = gradientDescent(X, y, theta, alpha, num_iters)
%GRADIENTDESCENT Performs gradient descent to learn theta
%   theta = GRADIENTDESENT(X, y, theta, alpha, num_iters) updates theta by 
%   taking num_iters gradient steps with learning rate alpha

% Initialize some useful values
m = length(y); % number of training examples
J_history = zeros(num_iters, 1);

for iter = 1:num_iters
    % ====================== YOUR CODE HERE ======================
    % Instructions: Perform a single gradient step on the parameter vector
    %               theta. 
    %
    % Hint: While debugging, it can be useful to print out the values
    %       of the cost function (computeCost) and gradient here.
    %
    summand = 0;
    for i = 1:m
       summand = summand + (X(i,:)*theta - y(i)).*X(i,:)';
    endfor

    theta = theta - alpha*(m)^-1*summand;
    % ============================================================

%    if (size(computeCost(X, y, theta)) ~= size(iter))
%       disp('Dims summand'), size(summand)
%       disp('Dims computeCost'), size(computeCost(X, y, theta))
%       computeCost(X,y,theta)
%       disp('\n dims(iter)'), size(iter)
%       disp('Size X, y, theta'), size(X), size(y), size(theta)
%    endif
    % Save the cost J in every iteration    
    J_history(iter) = computeCost(X, y, theta);
endfor
end
