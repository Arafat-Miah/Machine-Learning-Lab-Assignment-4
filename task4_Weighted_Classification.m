function w = trainPerceptronWeighted(X,y,betas)

    % << IMPLEMENT THE FUNCTION BODY! SAMPLE SOLUTION STRUCTURE SHOWN IN THE COMMENTS BELOW. >>

    % Initialize variables and set up the problem: initial parameters, cost function etc.

    % Solve the problem using gradientDescentAD
    
    % If necessary, define nested helper functions below
%   Inputs:
%       X: P x N feature matrix 
%       y: P x 1 label vector (-1 or 1)
%       betas: P x 1 vector of weights 
%
%   Output:
%       w: 1 x (N+1) trained weight vector

    % 1. Augment the Feature Matrix
    [P, N] = size(X);
    X_aug = [ones(P, 1), X];

    % 2. Hyperparameters
    alpha = 1.0; 
    max_iter = 5000;

    % 3. Initialize Weights
    w0 = zeros(1, N + 1);

    % 4. Define the Cost Function
    costFunc = @(w) weightedSoftmaxCost(w, X_aug, y, betas);

    % 5. Optimize using gradientDescentAD
    [~, w, ~, ~] = gradientDescentAD(costFunc, w0, alpha, max_iter);

    % ---------------------------------------------------------
    % NESTED HELPER FUNCTION: Weighted Softmax Cost
    % ---------------------------------------------------------
    function cost = weightedSoftmaxCost(W, X_aug, y, betas)
        % Inputs:
        %   W:     K x (N+1) matrix of weight candidates
        %   X_aug: P x (N+1) matrix of augmented data
        %   y:     P x 1 vector of labels
        %   betas: P x 1 vector of sample weights
        
        % Get the number of samples P from the data matrix
        P_samples = size(X_aug, 1);
        
        % 1. Calculate linear scores: z = X * w' 
        scores = X_aug * W';
        
        % 2. Calculate Softmax Log-Loss term: log(1 + exp(-y * z))
        log_loss = log(1 + exp(-y .* scores));
        
        % 3. Apply Weights (Betas)
        weighted_loss = betas .* log_loss;
        
        % 4. Compute Weighted Sum and Scale by 1/P
        total_weighted_sum = sum(weighted_loss, 1);
        
        % Scale by 1/P and transpose to return a K x 1 column vector
        cost = (1 / P_samples) * total_weighted_sum';
    end

end
