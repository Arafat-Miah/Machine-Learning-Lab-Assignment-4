function w = trainPerceptron(X,y)

    % << IMPLEMENT THE FUNCTION BODY! SAMPLE SOLUTION STRUCTURE SHOWN IN THE COMMENTS BELOW. >>

    % Initialize variables and set up the problem: initial parameters, cost function etc.

    % Solve the problem using gradientDescentAD
    
    % If necessary, define nested helper functions below


    

    % 1.Initialize variables and set up the problem
    [P, N] = size(X);
    
    % Augment X with a column of ones at the start for the bias term (x0 = 1)
    % X becomes P x (N+1)
    X_aug = [ones(P, 1), X];
    
    % 2. Hyperparameters
    lambda = 0.001; 
    
    % Learning rate (alpha) and max iterations
    alpha = 1.0; 
    max_iter = 5000;
    
    % 3. Initial weights
    % Initialize as a row vector of zeros [1 x (N+1)]
    w0 = zeros(1, N + 1);
    
    % 4. Define the Cost Function
    % I create a function handle 'g' that calls the nested helper function.
    g = @(w) regularizedSoftmaxCost(w, X_aug, y, lambda);

    % 5. Solve the problem using gradientDescentAD
    [~, w, ~, ~] = gradientDescentAD(g, w0, alpha, max_iter);

    
    function cost = regularizedSoftmaxCost(W, X_aug, y, lambda)
        % Inputs:
        %   W:     K x (N+1) matrix of weight candidates (rows are weight vectors)
        %   X_aug: P x (N+1) matrix of augmented data samples
        %   y:     P x 1 vector of labels (-1 or 1)
        %   lambda: Scalar regularization parameter
        % Output:
        %   cost:  K x 1 column vector of costs
        
        % Calculate the linear score: z = w^T * x
        % X_aug * W' results in a P x K matrix (samples x weight_candidates)
        scores = X_aug * W';
        
        % Calculate the exponent term: -y * z
        % I use element-wise multiplication with broadcasting. 
        % y is Px1, scores is PxK.
        exponent_term = -y .* scores;
        
        % Now Compute Softmax Loss: log(1 + exp(-y * z))
        loss = log(1 + exp(exponent_term));
        
        % Average the loss over all P samples (mean along columns)
        % Result is 1 x K
        mean_loss = mean(loss, 1);
        
        % Regularization: lambda * ||w||^2
        % I sum the squared weights along dimension 2 (the features)
        % Result is K x 1
        reg_term = lambda * sum(W.^2, 2);
        
        % Combine components
        % Transpose mean_loss to match reg_term (K x 1)
        cost = mean_loss' + reg_term;
    end
end
