data_trn = load('hw3_train.dat');
data_tst = load('hw3_test.dat');

rng(20121228);

N = size(data_trn, 1);
M = size(data_tst, 1);

tst_data = data_tst;
tst_size = M;

T = 300;
I = 100;
err_table = zeros(I, T);
err_table_forest = zeros(I, 1);

for iter = 1:I

y_forest = zeros(tst_size, 1);
y = zeros(tst_size, 1);
for t = 1:T
    forest_table(i, t).id = -1;
    % Bagging
    ix = randi(N, N, 1);
    data_bag = data_trn(ix, :);
    
    root = find_min_gini(data_bag);
    root.id = 1;
    
    tree = build_CART_tree(root);
    forest_table(i, t).tree = tree;
    
    % Compute E_in for the current tree

    for i = 1:tst_size
        y(i, 1) = predict_CART_tree(tree, tst_data(i, :));
        y_forest(i, 1) = y_forest(i, 1) + y(i, 1);
    end
    
    err = sum(y ~= tst_data(:, end)) / tst_size;
    err_table(iter, t) = err;
    
    % Compute E_out for the current tree

end
err_table_forest(iter, 1) = sum(sign(y_forest) ~= tst_data(:, end)) / tst_size;
fprintf('Done %03d\n', iter)
end

err_mean = mean(mean(err_table));
err_mean_forest = mean(err_table_forest);



