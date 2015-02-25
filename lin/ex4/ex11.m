%%%%%%%%%%%%%%%%%
%% Initilization
%%%%%%%%%%%%%%%%%

data_trn = load('hw4_nnet_train.dat');
X_trn = data_trn(:, 1:end-1);
y_trn = data_trn(:, end);

data_tst = load('hw4_nnet_test.dat');
X_tst = data_tst(:, 1:end-1);
y_tst = data_tst(:, end);

[N, d] = size(X_trn);
[K, d] = size(X_tst);

% Patch the X_trn
X_trn = [ones(N, 1) X_trn];
X_tst = [ones(K, 1) X_tst];

% Initialization
eta = 10;
r = 0.1;

M = 3;

rng(20121228);

T = 50000;
I = 500;

err_mat = zeros(I, 1);

for ix = 1:I
    % First layer (d+1)xM
    W1 = 2*r*rand(d+1, M) - r;
    
    % Second layer (M+1)x1
    W2 = 2*r*rand(M+1, 1) - r;
    
    for t = 1:T
        n = randi(N, 1, 1);
        
        %%%%%%%%%%%%%%%%%%%%%%%%
        %% Forward propogation
        %%%%%%%%%%%%%%%%%%%%%%%%
        % (d+1)x1
        X0 = X_trn(n, :)';
        
        % Mx(d+1) by (d+1)x1 = Mx1
        S1 = W1' * X0;
        X1 = [1; tanh(S1)];
        
        % 1x(M+1) by (M+1)x1 = 1x1
        S2 = W2' * X1;
        X2 = tanh(S2);
        
        %%%%%%%%%%%%%%%%%%%%%%%%
        %% Backward propogation
        %%%%%%%%%%%%%%%%%%%%%%%%
        % 1x1
        Delta2 = -2*(y_trn(n)-X2)*(1-tanh(S2).^2);
        
        % (M+1)x1 by 1x1 = (M+1)x1
        Delta1 = W2 * Delta2;
        
        Delta1 = Delta1(2:end, 1);
        % Mx1 dot Mx1 = Mx1
        Delta1 = Delta1 .* (1-tanh(S1).^2);
        
        %%%%%%%%%%%%%%%%%%%%%%%%
        %% Gradient decent
        %%%%%%%%%%%%%%%%%%%%%%%%
        % W2: (M+1)x1, X1*Delta2: (M+1)x1
        W2 = W2 - eta*X1*Delta2';
        % W1: (d+1)xM, X0*Delta11: (d+1)x1 by 1xM
        W1 = W1 - eta*X0*Delta1';
        
    end
    
    %%%%%%%%%%%%%%%%%%%%%%%%
    %% Test
    %%%%%%%%%%%%%%%%%%%%%%%%
    K = size(X_tst, 1);
    
    S1 = W1' * X_tst';
    X1 = [ones(1, K); tanh(S1)];
    S2 = W2' * X1;
    X2 = tanh(S2);
    
    y_pred = sign(X2');
    error = sum(y_pred ~= y_tst) / K;
    
    err_mat(ix) = error;
    fprintf('Done %d, %f\n', ix, error);

end

avg_error = mean(err_mat);
fprintf('avg error: %f\n', avg_error);