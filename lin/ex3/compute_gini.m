function gini = compute_gini(data)

N = size(data, 1);
y = data(:, end);
if N == 0
    gini = 0;
else
    gini = 1 - (sum(y == 1)/N)^2 - (sum(y == -1)/N)^2;

end