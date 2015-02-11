data_trn = load('hw3_train.dat');
X_trn = data_trn(:, 1:2);
y_trn = data_trn(:, end);

X_pos = data_trn(y_trn > 0, 1:2);
X_neg = data_trn(y_trn < 0, 1:2);

plot(X_pos(:, 1), X_pos(:, 2), 'bo');
hold on
plot(X_neg(:, 1), X_neg(:, 2), 'rx');

root = find_min_gini(data_trn);
root.id = 1;

tree = [root];
empty_node.id = -1;

root = build_CART_tree(root);

data_tst = load('hw3_test.dat');
N = size(data_tst, 1);
y = ones(N, 1);
for i = 1:N
    y(i, 1) = predict_CART_tree(root, data_tst(i, :));
end

err = sum(y ~= data_tst(:, end));





