function [y_pred, err] = AdaBoostPred(X, y, s, i, theta)

x = X(:, i);
y_pred = zeros(size(y));
y_pred(x < theta) = sign(-1 * s);
y_pred(x >= theta) = sign(1 * s);

err = sum(y ~= y_pred);

end