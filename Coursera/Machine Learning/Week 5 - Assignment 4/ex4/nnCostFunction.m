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

% You need to return the following variables correctly
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m

K =2; % Number of layers
for s = 1:m
   % sigmoid(Theta1*[0;X(s,:).']).' is the a2 column vector
   % sigmoid(Theta2*a2) is the a3 column vector
   h_theta = sigmoid(Theta2*[1;sigmoid(Theta1*[1;X(s,:).'])]);
   % sparse(1,y(s),1,1,num_labels) makes a 1xnum_labels row vector with a 1 in the 1xy(s) spot
   J = J - (sparse(1,y(s),1,1,num_labels)*log(h_theta) + ...
   (1-sparse(1,y(s),1,1,num_labels))*log(1-h_theta));
   ;
end
% scale the error by the number of samples and regularize the cost
J = m^-1*( J + ...
.5*lambda*(sum(sum(Theta1(:,2:end).^2))+sum(sum(Theta2(:,2:end).^2))));

%
% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the
%               first time.

%keyboard;
Delta1 = 0;
Delta2 = 0;
for t = 1:m
   z2 = Theta1*[1;X(t,:).'];
   a2 = sigmoid(z2);
   a2 = [1;a2]; % add bias unit
   z3 = Theta2*a2;
   a3 = sigmoid(z3);
   delta3 = a3 - sparse(1,y(t),1,1,num_labels).';
%   keyboard;
   delta2 = Theta2.'*delta3.*[1;sigmoidGradient(z2)];
   Delta1 = Delta1 + delta2(2:end)*[1,X(t,:)];
   Delta2 = Delta2 + delta3*a2.';
end

% The complicated second term makes sure to turn the 1st column of Theta1 into
% zeros
Theta1_grad=Delta1/m+(lambda/m)*[zeros(1,hidden_layer_size);Theta1(:,2:end).'].';
% The complicated second term makes sure to turn the 1st column of Theta1 into
% zeros
Theta2_grad=Delta2/m+(lambda/m)*[zeros(1,num_labels);Theta2(:,2:end).'].';

%
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.



% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];

end
