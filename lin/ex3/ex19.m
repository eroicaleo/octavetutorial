data_trn = load('hw3_train.dat');
data_tst = load('hw3_test.dat');

rng(20121228);

N = size(data_trn, 1);
M = size(data_tst, 1);
T = 300;
I = 100;

err_table = zeros(I, T);
err_forst = zeros(I, 1);
err_table_tst = zeros(I, T);
err_forst_tst = zeros(I, 1);

y_trn = data_trn(:, end);
y_tst = data_tst(:, end);

for iter = 1:I
    
    y_forest = zeros(N, 1);
    y_forest_tst = zeros(M, 1);
    for t = 1:T
  
        % Bagging
        ix = randi(N, N, 1);
        data_bag = data_trn(ix, :);
        
        root = find_min_gini(data_bag);
        root.id = 1;
        
        % E_in
        y_pred = predict_pruned_tree(root, data_trn);
        y_forest = y_forest + y_pred;
        err_table(iter, t) = sum(y_pred ~= y_trn) / N;
        
        % E_out
        y_pred = predict_pruned_tree(root, data_tst);
        y_forest_tst = y_forest_tst + y_pred;
        err_table_tst(iter, t) = sum(y_pred ~= y_tst) / M;
                
    end
    
    err_forst(iter) = sum(sign(y_forest) ~= y_trn) / N;
    err_forst_tst(iter) = sum(sign(y_forest_tst) ~= y_tst) / M;
    fprintf('Done %03d\n', iter);
end

err_forst_mean = mean(err_forst);
err_forst_tst_mean = mean(err_forst_tst);