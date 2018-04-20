function [J, grad] = linearRegCostFunction(X, y, theta, lambda)
%LINEARREGCOSTFUNCTION Compute cost and gradient for regularized linear 
%regression with multiple variables
%   [J, grad] = LINEARREGCOSTFUNCTION(X, y, theta, lambda) computes the 
%   cost of using theta as the parameter for linear regression to fit the 
%   data points in X and y. Returns the cost in J and the gradient in grad

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost and gradient of regularized linear 
%               regression for a particular choice of theta.
%
%               You should set J to the cost and grad to the gradient.
%

m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;

h_x=X*theta;
err=h_x-y;
err_sq=err.^2;
theta_sq = theta.^2;
J=((1/(2*m)) * sum(err_sq)); % + ((lambda/(2*m)) * theta_sq);
grad = (X'*err)*(1/m);

theta(1) = 0;
theta_s= (lambda/(2*m)) * sum(theta.^2);
J=J+theta_s;
grad = grad + ((lambda/m) * theta);


%J_history = zeros(num_iters, 1);
%
%for iter = 1:num_iters
%    h_x=X*theta;
%    err=h_x-y;
%
%    change=(X'*err)*((1/m)*alpha)
%    theta=theta-change;
%
%    J_history(iter) = computeCost(X, y, theta);
%
%end
%










% =========================================================================

grad = grad(:);

end
