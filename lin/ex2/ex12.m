clear all

trn = load('hw2_adaboost_train.dat');
N = size(trn, 1);

X = trn(:, 1:end-1);
y = trn(:, end);

T = 300;
u_vec = zeros(N, T);
err_t_vec = zeros(1, T);
alpha_vec = zeros(1, T);
g_vec = zeros(3, T);

u = 1/N * ones(N, 1);
y_pred_G = zeros(size(y));

t = 1;
err_N_trn = zeros(1, T);
err_N_tst = zeros(1, T);

for t = 1:T
u_vec(:, t) = u;

[s, i, theta, err, y_pred] = AdaBoostTrain(X, y, u);
g_vec(:, t) = [s, i, theta]';

err_t = u'*(y_pred ~= y) / sum(u);
diamond = sqrt((1-err_t)/err_t);
err_t_vec(t) = err_t;

u(y_pred ~= y) = u(y_pred ~= y) * diamond;
u(y_pred == y) = u(y_pred == y) / diamond;

alpha = log(diamond);
alpha_vec(t) = alpha;

y_pred_G = y_pred_G + alpha * y_pred;
y_pred_t = sign(y_pred_G);
err_N_trn(1, t) = sum(y_pred_t ~= y);

end

u_sum = sum(u_vec);
min_err_t = min(err_t_vec);

y_pred_G = zeros(size(y));
for t = 1:T
[y_pred err] = AdaBoostPred(X, y, g_vec(1, t), g_vec(2, t), g_vec(3, t));
y_pred_G = y_pred_G + alpha_vec(t) * y_pred;
y_pred_t = sign(y_pred_G);
err_N_tst(1, t) = sum(y_pred_t ~= y);

end

tst = load('hw2_adaboost_test.dat');
N = size(tst, 1);

X = tst(:, 1:end-1);
y = tst(:, end);
y_pred_G = zeros(size(y));
err_N_tst = zeros(1, T);

for t = 1:T
[y_pred err] = AdaBoostPred(X, y, g_vec(1, t), g_vec(2, t), g_vec(3, t));
y_pred_G = y_pred_G + alpha_vec(t) * y_pred;
y_pred_t = sign(y_pred_G);
err_N = sum(y_pred_t ~= y);
err_t = sum(y_pred_t ~= y_pred);
err_N_tst(t) = err_N;

end

err_N_tst(end);










