function y_pred = predict_pruned_tree(root, data)

N = size(data, 1);
y_pred = zeros(N, 1);

f = root.feature;
lvote = sign(sum(root.data_l(:, end)));
rvote = sign(sum(root.data_r(:, end)));

lix = (data(:, f) <= root.theta);
rix = (data(:, f) >  root.theta);

y_pred(lix) = lvote;
y_pred(rix) = rvote;

end