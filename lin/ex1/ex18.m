% See the FAQ:
% Q: How could I generate the primal variable w of linear SVM?

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

for C = [0.001 0.01 0.1 1 10]
  libsvm_options = sprintf('-t 2 -g 100 -c %f', C)
  model = svmtrain(y, X, libsvm_options);
  
  % Compute w^T * w
  % w = model.SVs' * model.sv_coef;

  predicted_label = svmpredict(y_tst, X_tst, model);
  err = sum(predicted_label != y_tst);
  fprintf('Error for C: %f is %d\n', C, err);

  % Compute sum(xi)
  sum(abs(model.sv_coef) == C)

  input('Press any key to continue!');
end
