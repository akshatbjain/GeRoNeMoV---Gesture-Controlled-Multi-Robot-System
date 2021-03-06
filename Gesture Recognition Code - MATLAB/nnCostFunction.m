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
         
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

%  Feedforward the neural network and return the cost in the variable J.
%  Implement the backpropagation algorithm to compute the gradients
%  Theta1_grad and Theta2_grad. We should return the partial derivatives of
%  the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%  Theta2_grad, respectively.
%  Implement regularization with the cost function and gradients.

a1 = X;
a1 = [ones(size(X,1),1) X];

z2 = a1*Theta1';
a2 = sigmoid(z2);
a2 = [ones(size(a1, 1), 1) a2];

z3 = a2*Theta2';
a3 = sigmoid(z3);
hypo = a3;

y2 = zeros(size(hypo));
for i = 1:m
    y2(i,y(i)) = 1;
end

for i = 1:m
    for k = 1:num_labels
        J = J + ((-y2(i,k)*log(hypo(i,k))) - ((1-y2(i,k))*log(1-hypo(i,k))));
    end
end

sum1 = 0;
sum2 = 0;
for j = 1:size(Theta1, 1)
    for k = 2:size(Theta1,2)
        sum1 = sum1 + Theta1(j,k)^2;
    end
end

for j = 1:size(Theta2, 1)
    for k = 2:size(Theta2,2)
        sum2 = sum2 + Theta2(j,k)^2;
    end
end

sum = sum1 + sum2;
sum = (lambda/2)*sum;

J = J + sum;
J = J/m;

delta1 = 0;
delta2 = 0;

%Commencing Backpropagation Algorithm
for i  = 1:m
    a1 = X(i,:);            %   1x400
    a1 = [1 a1];            %   1x401
    z2 = a1*Theta1';        %   1x401 * 401x25 = 1x25
    a2 = sigmoid(z2);       %   1x25
    a2 = [1 a2];            %   1x26
    z3 = a2*Theta2';        %   1x26 * 26x10 = 1x10
    a3 = sigmoid(z3);       %   1x10
    
    del3 = a3 - y2(i,:);      %   1x10
    del2 = del3*Theta2.*sigmoidGradient([1 z2]);     %  1x10 * 10x26 .* 1x26 = 1x26
    del2 = del2(2:end);
    delta2 = delta2 + del3'*a2;  %   10x1 * 1x26 = 10x26
    delta1 = delta1 + del2'*a1;  %   25x1 * 1x401 = 25x401
end

theta_1 = Theta1;
theta_2 = Theta2;
theta_1(:,1) = 0;
theta_2(:,1) = 0;

Theta1_grad = delta1 + lambda*theta_1;         %25x401
Theta2_grad = delta2 + lambda*theta_2;         %10x26

Theta1_grad = Theta1_grad/m;         %25x401
Theta2_grad = Theta2_grad/m;         %10x26

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
