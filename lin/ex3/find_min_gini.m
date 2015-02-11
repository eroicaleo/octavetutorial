function node = find_min_gini(data)

% y = data(:, 3);
% X_pos = data(y > 0, 1:2);
% X_neg = data(y < 0, 1:2);
% 
% close all
% figure(1)
% plot(X_pos(:, 1), X_pos(:, 2), 'bo');
% hold on
% plot(X_neg(:, 1), X_neg(:, 2), 'rx');

min_gini = inf;

% Assume the data has the format  [x1, x2, y]
for feature = 1:size(data, 2)-1
    x_sorted_data = data(:, feature);
    x_top = unique(sort(x_sorted_data));
    x_bot = [-inf; x_top(1:end-1)];
    theta = (x_bot + x_top)/2;
    
    for s = theta'
        data_l = data(data(:, feature) < s, :);
        data_r = data(data(:, feature) > s, :);
        size_l = size(data_l, 1);
        size_r = size(data_r, 1);
        gini_l = compute_gini(data_l);
        gini_r = compute_gini(data_r);
        gini = size_l * gini_l + size_r * gini_r;
        if gini < min_gini
            node.feature = feature;
            node.theta = s;
            node.gini = gini;
            node.data_l = data_l;
            node.data_r = data_r;
            min_gini = gini;
        end
    end

end

end