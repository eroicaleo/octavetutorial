data = load('two_phase.dat');

X = data(:, 1:4);
y = data(:, 10);

phase2_power = [2.2, 2.4, 2.6, 2.8, 3.0, 3.2, 3.4, 3.6, 3.8]';
% phase2_power = [3.2:0.2:3.8]';


figure(1);
for i = 1:size(phase2_power, 1)
idx = (X(:, 2) == phase2_power(i, 1));

X_trn = X(idx, :);
y_trn = y(idx, :);

X_pos = X_trn(y_trn > 0, :);
X_neg = X_trn(y_trn < 0, :);

subplot(3, 3, i)
plot(X_pos(:, 3), X_pos(:, 4), 'bo');
hold on
plot(X_neg(:, 3), X_neg(:, 4), 'rx');
xlabel('Phase 1 power (W)');
ylabel('Phase 2 power (W)');
title(sprintf('Phase 2 charge = %.1fW', phase2_power(i)));
% legend('HPWF', 'LPWF');
end



