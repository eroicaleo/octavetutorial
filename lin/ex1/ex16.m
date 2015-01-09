% See the FAQ:
% Q: How could I generate the primal variable w of linear SVM?

addpath('/home/yang/programming/octavetutorial/libsvm-3.20/matlab');

clear all
trn = load('features.train');

y = trn(:, 1);
X = trn(:, 2:end);

N = size(X, 1);

for label = [0 2 4 6 8]
  y = trn(:, 1);
  y(y != label) = -1;
  y(y == label) = 1;
  
  pos = (y == 1);
  X_pos = X(pos, :);
  
  neg = (y == -1);
  X_neg = X(neg, :);
  
  % trn(1:10, :)
  % y(1:10, :)
  figure(label+1);
  plot(X_pos(:, 1), X_pos(:, 2), 'rx');
  hold on;
  plot(X_neg(:, 1), X_neg(:, 2), 'bo');
  
  libsvm_options = '-d 2 -t 1 -g 1 -r 1 -c 0.01'
  model = svmtrain(y, X, libsvm_options);
  
  predicted_label = svmpredict(y, X, model, 'libsvm_options');
  err = sum(predicted_label != y);
  fprintf('Error for label: %d is %d\n', label, err);
  % problem 17
  s = sum(y(model.sv_indices).*model.sv_coef);
  fprintf('Summation for alpha: %d is %f\n', label, s);
  % trn(model.sv_indices(1:10), 1)
  % trn(model.sv_indices(model.sv_coef < 0)(1:10), 1)
  input('Press any key to continue!');
end
