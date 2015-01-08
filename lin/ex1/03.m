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

N = size(trn, 1);

% Q = [q_nm] = [y_n * y_m * K(x_n, x_m)]
Knm = X*X'+1;
Knm = Knm.^2;
Q = y*y' .* Knm
H = Q;

% p = [-1]
p = -ones(N, 1)
q = p;

% \sum (y_n * \alpha_n) = 0
A = y';
b = 0;

% \alpha_n >= 0;
lb = zeros(N, 1);
ub = [];

alpha0 = zeros(N, 1);
[alpha, obj, info, lambda] = qp (alpha0, H, q, A, b, lb, ub);
obj
alpha
sum(alpha)

addpath('/home/yang/programming/octavetutorial/libsvm-3.20/matlab');
libsvm_options = '-t 1 -g 1 -r 1 -d 2 -c 1000';
model = svmtrain(y, X, libsvm_options);
w = model.SVs' * model.sv_coef;
b = -model.rho;

if model.Label(1) == -1
w = -w;
b = -b;
end

model.sv_coef
sum(abs(model.sv_coef))
model.sv_indices
w
b

%% problem 4
%w = sum(a_n * y_n * z_n)
%z_n = (1, x1, x2, x1^2, x1x2, x2^2)
x1 = X(:, 1);
x2 = X(:, 2);
zn = [ones(N, 1) sqrt(2)*x1 sqrt(2)*x2 x1.^2 x1.*x2 x2.^2]
a = y .* alpha;
w = a' * zn

% y2 is a support vector
% b = y2 - sum(an * yn * K(n, 2))
Kn2 = X*X(2, :)' + 1;
Kn2 = Kn2.^2;
b = y(2) - sum(alpha .* y .* Kn2)

hold on
ezplot(@(x1,x2) (15 + 16*x1 - 8*x1.^2 - 6*x2.^2))

%% problem 5
%% plot problem 2
hold on
ezplot(@(x1,x2) (x2.^2 - 2*x1 + 3 - 4.5))
