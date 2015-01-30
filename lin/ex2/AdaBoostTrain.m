function [s, i, theta, err, y_pred] = AdaBoostTrain(X, y, u)

[N, d] = size(X);
err_min = u' * ones(N, 1);
y_pred = zeros(size(y));

for i = 1:d
  x = X(:, i);
  [x_sorted IX] = sort(x);
  % append the -inf element at the front
  x_prefix = [-inf; x_sorted];
  for n = 1:N
    theta = (x_prefix(n)+x_prefix(n+1))/2;
    for s = [-1, 1]
      y_pred(x < theta) = -1 * s;
      y_pred(x >= theta) = 1 * s;
      err = sum( u'*(y_pred ~= y) );
      if (err < err_min)
        err_min = err;
        res = [s, i, theta];
      end
    end
  end
end

s = res(1);
i = res(2);
theta = res(3);
err = err_min;
[y_pred err_pred] = AdaBoostPred(X, y, s, i ,theta);

end