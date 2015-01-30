data = load('hw2_lssvm_all.dat');

trn = data(1:400, :);
tst = data(401:end, :);

X_trn = trn(:, 1:end-1);
y_trn = trn(:, end);

X_tst = tst(:, 1:end-1);
y_tst = tst(:, end);

X_prod = X_trn .* X_trn;
X_prod = sum(X_prod, 2);

X_prod = repmat(X_prod, 1, 400);

X_cros = X_trn * X_trn';

X_dist = X_prod - 2*X_cros + X_prod';

t12 = sum((X_trn(1, :) - X_trn(2, :)).^2);
t13 = sum((X_trn(1, :) - X_trn(3, :)).^2);
t23 = sum((X_trn(2, :) - X_trn(3, :)).^2);

X_trn_prod = sum(X_trn .* X_trn, 2);
X_trn_prod_rep = repmat(X_trn_prod, 1, 100);
X_tst_prod = sum(X_tst .* X_tst, 2);
X_tst_prod_rep = repmat(X_tst_prod, 1, 400);
X_cros = X_trn * X_tst';

X_dist_trn_tst = X_trn_prod_rep - 2*X_cros + X_tst_prod_rep';
t12 = sum((X_trn(1, :) - X_tst(2, :)).^2);
t13 = sum((X_trn(1, :) - X_tst(3, :)).^2);
t23 = sum((X_trn(2, :) - X_tst(3, :)).^2);


gamma = [32 2 0.125];
lambda = [0.001, 1, 1000];

for g = gamma
    for l = lambda
        K = exp(-g * X_dist);
        K_tst = exp(-g * X_dist_trn_tst);
        beta = (l*eye(400)+K) \ y_trn;
        Ein_pred = sign(beta' * K)';
        err_trn = sum(Ein_pred ~= y_trn);
        Eout_pred = sign(beta' * K_tst)';
        err_tst = sum(Eout_pred ~= y_tst);
        fprintf('gamma = %.4f, lambda = %.4f, Ein_pred = %.2f, Eout_pred = %.2f\n', ...
            g, l, err_trn/sum(abs(y_trn)), err_tst/sum(abs(y_tst)) );
    end
end






