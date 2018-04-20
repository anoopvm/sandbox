function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

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

h = sigmoid(X * theta);

red = ((-1) * y)' * log(h);
blue = (repmat(1, size(y)) - y)' * log(repmat(1, size(h)) - h);
J1 = (1/m) * (red - blue);

grad1 = (1/m) * (X' * (h-y))
theta(1) = 0;
thetasq = theta' * theta;
reg = (lambda / (2*m)) * thetasq;
reg2 = (lambda / m) * theta; 
J = J1 + reg; 
grad = grad1 + reg2;




% =============================================================

end
