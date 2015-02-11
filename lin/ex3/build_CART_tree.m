function node = build_CART_tree(node)

% termination condition
sz = min(size(node.data_l, 1), size(node.data_r, 1));
if sz == 0
    return
end

% fprintf('Node: %d\n', node.id);

lnode = find_min_gini(node.data_l);
lnode.id =  2*node.id;
lnode = build_CART_tree(lnode);
node.lnode = lnode;

rnode = find_min_gini(node.data_r);
rnode.id = 2*node.id+1;
rnode = build_CART_tree(rnode);
node.rnode = rnode;

end