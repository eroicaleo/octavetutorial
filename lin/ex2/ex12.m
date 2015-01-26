clear all

trn = load('hw2_adaboost_train.dat');
N = size(trn);

X = trn(:, 1:end-1);
y = trn(:, end);

d = size(X, 2);

err_min = N;
for i = 1:d
  x = X(:, i);
  [x_sorted IX] = sort(x);
  y_sorted = y(IX);
  tmp = [x_sorted y_sorted];
  % append the -inf element at the front
  x_prefix = [x_sorted(1, 1)-1; x_sorted];
  for n = 1:N
    theta = (x_prefix(n)+x_prefix(n+1))/2;
    for s = [-1, 1]
      y_pred = s*ones(N+1, 1);
      y_pred(1:n) = s*-1;
      y_prefix = [s*-1; y];
      err = sum(y_pred != y_prefix);
      if (err < err_min)
        err_min = err;
	res = [i, n, theta]
      end
    end
  end
end
