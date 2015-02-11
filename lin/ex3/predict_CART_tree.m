function y = predict_CART_tree(node, data)


f = node.feature;
lsize = size(node.data_l, 1);
rsize = size(node.data_r, 1);

% termination condition
if lsize == 0
    y = node.data_r(1, end);
    return
elseif rsize == 0
    y = node.data_l(1, end);
    return
end

if (data(1, f) <= node.theta)
    % branch left
    y = predict_CART_tree(node.lnode, data);
elseif (data(1, f) > node.theta)
    % branch right
    y = predict_CART_tree(node.rnode, data);
end

end