addpath('/home/yang/programming/octavetutorial/libsvm-3.20/matlab');

clear all
trn = load('features.train');
tst = load('features.test');

y = trn(:, 1);
X = trn(:, 2:end);

y_tst = tst(:, 1);
X_tst = tst(:, 2:end);

N = size(X, 1);

y(y != 0) = -1;
y(y == 0) = 1;

y_tst(y_tst != 0) = -1;
y_tst(y_tst == 0) = 1;

trn(1:10, :)
y(1:10, :)

C = 0.1;
for gamma = [1 10 100 1000 10000]
  libsvm_options = sprintf('-t 2 -g %f -c 0.1', gamma)
  model = svmtrain(y, X, libsvm_options);
  
  predicted_label = svmpredict(y_tst, X_tst, model);
  err = sum(predicted_label != y_tst);
  fprintf('Error for gamma: %f is %d\n', gamma, err);

  % Compute sum(xi)
  sum(abs(model.sv_coef) == C)

  input('Press any key to continue!');
end
