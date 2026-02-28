function theta = fitNonlinearSoftmax( X, y )

    % << IMPLEMENT THE FUNCTION BODY! TYPICAL STEPS ARE GIVEN IN COMMENTS BELOW >>

    % Initialize variables
    theta0 = zeros(1, 3);
    alpha = 1.0;
    max_iter = 5000;
    % Solve the weights using GD with AD, call [...] = gradientDescentAD(...)
    [~, theta, ~, ~] = gradientDescentAD(@cost, theta0, alpha, max_iter);
    % Return the best solution, and the histories
    
    % This function computes the Softmax cost function on nonlinear model
    % NOTE: As a nested function, it can use X and y directly and needs only the parameter vector theta
    function c = cost(theta)
        % Calculate the model output
        scores = model(X, theta);
        
        % Calculate Softmax Log-Loss: log(1 + exp(-y * model_out))
        % I use -y .* scores for element-wise broadcasting
        log_loss = log(1 + exp(-y .* scores));
        
        % Average the loss over all samples 
        % Return as a column vector 
        c = mean(log_loss, 1)';
    end

end

% Local helper functions below

% This function transforms the features X non-linearly using the parameters v
function z = feature_transform( X, v )
    % No parameters needed for this transform so v not used!
    z = X.^2;% << COMPUTE THE TRANSFORM >>
end

% This function applies the model specified by the parameters theta to the data X
function y = model(X, theta)
    % 1. Transform features: x -> f(x) = [x1^2, x2^2]
        F = feature_transform(X, []);
        
        % 2. Augment with bias term (column of ones)
        [P, ~] = size(F);
        F_aug = [ones(P, 1), F];
        
        % 3. Compute Dot Product: f_aug * theta'
        y = F_aug * theta';
end
