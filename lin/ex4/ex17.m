%%%%%%%%%%%%%%%%%
%% Initilization
%%%%%%%%%%%%%%%%%

data_trn = load('hw4_knn_train.dat');
X_trn = data_trn(:, 1:end-1);
y_trn = data_trn(:, end);

data_tst = load('hw4_knn_test.dat');
X_tst = data_tst(:, 1:end-1);
y_tst = data_tst(:, end);

[N, d] = size(X_trn);

[M, d] = size(X_tst);

%%%%%%%%%%%%%%%%%
%% Ein
%%%%%%%%%%%%%%%%%
y_prd = zeros(N, 1);

for i = 1:N
    x = X_trn(i, :);
    x_rep = repmat(x, N, 1);
    x_dist = x_rep - X_trn;
    % N by 1 
    x_dist = sum(x_dist .* x_dist, 2);
    [B IX] = sort(x_dist);
    y_knn = y_trn(IX);
    y_prd(i, 1) = sign(sum(y_knn(1:5, 1)));
end

error = sum(y_prd ~= y_trn) / N;
fprintf('avg error: %f\n', error);

%%%%%%%%%%%%%%%%%
%% Eout
%%%%%%%%%%%%%%%%%
y_prd = zeros(M, 1);

for i = 1:M
    x = X_tst(i, :);
    x_rep = repmat(x, N, 1);
    x_dist = x_rep - X_trn;
    % N by 1 
    x_dist = sum(x_dist .* x_dist, 2);
    [B IX] = sort(x_dist);
    y_knn = y_trn(IX);
    y_prd(i, 1) = sign(sum(y_knn(1:5, 1)));
end

error = sum(y_prd ~= y_tst) / M;
fprintf('avg error: %f\n', error);
