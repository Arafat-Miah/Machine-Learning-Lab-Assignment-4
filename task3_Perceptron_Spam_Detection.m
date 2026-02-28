function w = trainPerceptronNormalized(X,y)

    % << IMPLEMENT THE FUNCTION BODY! SAMPLE SOLUTION STRUCTURE SHOWN IN THE COMMENTS BELOW. >>

    % Initialize variables and set up the problem: initial parameters, cost function etc.

    % Solve the problem using gradientDescentAD
    
    % If necessary, define nested helper functions below

%   Task:
%   1. Normalize the features in X (zero mean, unit variance).
%   2. Train using Regularized Softmax cost function.
  
    % 1. Normalize the Feature Matrix X
    X_norm = normalize(X);

    % 2. Augment the Normalized Feature Matrix
    [P, N] = size(X_norm);
    X_aug = [ones(P, 1), X_norm];

    % 3. Hyperparameters
    lambda = 0.001; 
    
    % Learning rate (alpha) and max iterations
    alpha = 1.0; 
    max_iter = 5000;

    % 4. Initialize Weights
    % Initialize w0 as a row vector of zeros [1 x (N+1)]
    w0 = zeros(1, N + 1);

    % 5. Define the Cost Function.
    costFunc = @(w) regularizedSoftmaxCost(w, X_aug, y, lambda);

    % 6. Optimize using gradientDescentAD
    [~, w, ~, ~] = gradientDescentAD(costFunc, w0, alpha, max_iter);

    % ---------------------------------------------------------
    % NESTED HELPER FUNCTION: Regularized Softmax Cost
    % ---------------------------------------------------------
    function cost = regularizedSoftmaxCost(W, X_aug, y, lambda)
        % Inputs:
        %   W: K x (N+1) matrix of weight candidates
        %   X_aug: P x (N+1) matrix of augmented data
        %   y: P x 1 vector of labels (-1 or 1)
        
        % Calculate linear scores: z = X * w'
        % Result is P x K
        scores = X_aug * W';
        
        
        softmax_loss = log(1 + exp(-y .* scores));
        
        
        avg_loss = mean(softmax_loss, 1);
        
        reg_term = lambda * sum(W.^2, 2);
        cost = avg_loss' + reg_term;
    end

end

% Load X and y from 'spambase_data.mat'
load('spambase_data.mat');

% Call your training function
w = trainPerceptronNormalized(X,y);
