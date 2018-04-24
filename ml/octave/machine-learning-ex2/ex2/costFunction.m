function [J, grad] = costFunction(theta, X, y)
%COSTFUNCTION Compute cost and gradient for logistic regression
%   J = COSTFUNCTION(theta, X, y) computes the cost of using theta as the
%   parameter for logistic regression and the gradient of the cost
%   w.r.t. to the parameters.

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta
%
% Note: grad should have the same dimensions as theta
%

h = sigmoid(X * theta);

%red = ((-1) * eye(size(y))' * log(h));
red = ((-1) * y)' * log(h);
blue = (repmat(1, size(y)) - y)' * log(repmat(1, size(h)) - h);
J = (1/m) * (red - blue);

grad = (1/m) * (X' * (h-y))
%theta(1) = 0;
%thetasq = theta' * theta;
%reg = (lambda / (2*m));
%J = J1 + reg; 

% =============================================================

end
