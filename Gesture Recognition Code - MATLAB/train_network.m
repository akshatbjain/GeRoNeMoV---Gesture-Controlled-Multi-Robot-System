%% Initialization
clear ; close all; clc

%% Setting up the parameters we will use
input_layer_size  = 400;  % 20x20 Input Images
hidden_layer_size = 25;   % 25 hidden units
num_labels = 6;          % 6 labels, from 1 to 6   

%% =========== Loading and Visualizing Data =============
%  We start by first loading and visualizing the dataset. 
% Load Training Data
fprintf('Loading and Visualizing Data ...\n')

load('dataset3.mat');
y = y';
m = size(X_new, 3);
temp = X_new(:,:,1);
X = [temp(:)'];
for (i = 2:m)
    temp = X_new(:,:,i);
    X = [X;temp(:)'];
end
X = double(X);

% m = size(X, 1);

% Randomly select 100 data points to display
sel = randperm(size(X, 1));
sel = sel(1:100);

displayData(X(sel, :));

fprintf('Program paused. Press enter to continue.\n');
pause;


%% ================ Loading Parameters ================
% In this part, we load some pre-initialized 
% neural network parameters.

fprintf('\nLoading Saved Neural Network Parameters ...\n')

% Loading the weights into variables Theta1 and Theta2
load('ex4weights.mat');

% Unrolling parameters 
nn_params = [Theta1(:) ; Theta2(:)];


%% ================ Sigmoid Gradient  ================
%  Before we start implementing the neural network, we will first
%  implement the gradient for the sigmoid function.

fprintf('\nEvaluating sigmoid gradient...\n')

g = sigmoidGradient([1 -0.5 0 0.5 1]);
fprintf('Sigmoid gradient evaluated at [1 -0.5 0 0.5 1]:\n  ');
fprintf('%f ', g);
fprintf('\n\n');

fprintf('Program paused. Press enter to continue.\n');
pause;


%% ================ Initializing Pameters ================
%  In this part, we will be starting to implment a two
%  layer neural network that classifies patterns. We will start by
%  implementing a function to initialize the weights of the neural network
%  (randInitializeWeights.m)

fprintf('\nInitializing Neural Network Parameters ...\n')

initial_Theta1 = randInitializeWeights(input_layer_size, hidden_layer_size);
initial_Theta2 = randInitializeWeights(hidden_layer_size, num_labels);

% Unrolling parameters
initial_nn_params = [initial_Theta1(:) ; initial_Theta2(:)];


%% =============== Implement Backpropagation ===============

fprintf('\nChecking Backpropagation... \n');

%  Checking gradients by running checkNNGradients
checkNNGradients;

fprintf('\nProgram paused. Press enter to continue.\n');
pause;

%% =================== Training NN ===================
%  We have now implemented all the code necessary to train a neural 
%  network. To train our neural network, we will now use "fmincg". This is
%  an advanced optimizer that is able to train our cost functions efficiently
%  as long as we provide them with the gradient computations.

fprintf('\nTraining Neural Network... \n')

options = optimset('MaxIter', 200);

%  You can try different values of lambda
%  This worked best for us
lambda = 30;

% Creating "short hand" for the cost function to be minimized
costFunction = @(p) nnCostFunction(p, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, X, y, lambda);

% Now, costFunction is a function that takes in only one argument (the
% neural network parameters)
[nn_params, cost] = fmincg(costFunction, initial_nn_params, options);

% Obtaining Theta1 and Theta2 back from nn_params
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

fprintf('Program paused. Press enter to continue.\n');
pause;


%% ================= Visualize Weights =================
%  We can now "visualize" what the neural network is learning by 
%  displaying the hidden units to see what features they are capturing in 
%  the data.

fprintf('\nVisualizing Neural Network... \n')

displayData(Theta1(:, 2:end));

fprintf('\nProgram paused. Press enter to continue.\n');
pause;

%% ================= Implement Predict =================
%  After training the neural network, we would like to use it to predict
%  the labels. We will now implement the "predict" function to use the
%  neural network to predict the labels of the training set. This lets
%  us compute the training set accuracy.

pred = predict(Theta1, Theta2, X);

fprintf('\nTraining Set Accuracy: %f\n', mean(double(pred == y)) * 100);

save weights Theta1 Theta2


