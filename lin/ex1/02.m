clear all;
close all;

trn = [
1 0 -1;
0 1 -1;
0 -1 -1;
-1 0 1;
0 2 1;
0 -2 1;
-2 0 1;
];

X = trn(:, [1 2]);
y = trn(:, end);

pos = (y == 1);
X_pos = X(pos, :);

neg = (y == -1);
X_neg = X(neg, :);

figure(1);
plot(X_pos(:, 1), X_pos(:, 2), 'rx');
hold on;
plot(X_neg(:, 1), X_neg(:, 2), 'bo');

z1 = X(:, 2).^2 - 2*X(:, 1) + 3;
z2 = X(:, 1).^2 - 2*X(:, 2) - 3;

Z = [z1 z2];
Z_pos = Z(pos, :);
Z_neg = Z(neg, :);

figure(2);
plot(Z_pos(:, 1), Z_pos(:, 2), 'rx');
hold on;
plot(Z_neg(:, 1), Z_neg(:, 2), 'bo');


