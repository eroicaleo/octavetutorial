% See the FAQ:
% Q: How could I generate the primal variable w of linear SVM?

addpath('/home/yang/programming/octavetutorial/libsvm-3.20/matlab');

clear all
trn = load('features.train');

y = trn(:, 1);
X = trn(:, 2:end);

y(y != 0) = -1;
y(y == 0) = 1;

trn(1:10, :);
y(1:10, :);

libsvm_options = '-t 0 -c 0.01'
model = svmtrain(y, X, libsvm_options);
w = model.SVs' * model.sv_coef;
w
sqrt(w' * w)


