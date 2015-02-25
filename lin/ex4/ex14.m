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
eta = 0.01;
r = 0.1;

M1 = 8;
M2 = 3;

rng(20121228);

T = 50000;
I = 500;

err_mat = zeros(I, 1);

for ix = 1:I
    % First layer (d+1)xM
    W1 = 2*r*rand(d+1, M1) - r;
    
    % Second layer (M1+1)xM2
    W2 = 2*r*rand(M1+1, M2) - r;
    
    % Second layer (M2+1)x1
    W3 = 2*r*rand(M2+1, 1) - r;
    
    for t = 1:T
        n = randi(N, 1, 1);
        
        %%%%%%%%%%%%%%%%%%%%%%%%
        %% Forward propogation
        %%%%%%%%%%%%%%%%%%%%%%%%
        % (d+1)x1
        X0 = X_trn(n, :)';
        
        % M1x(d+1) by (d+1)x1 = M1x1
        S1 = W1' * X0;
        X1 = [1; tanh(S1)];
        
        % M2x(M1+1) by (M1+1)x1 = M2x1
        S2 = W2' * X1;
        X2 = [1; tanh(S2)];
        
        % 1x(M2+1) by (M2+1)x1 = 1x1
        S3 = W3' * X2;
        X3 = tanh(S3);
        
        %%%%%%%%%%%%%%%%%%%%%%%%
        %% Backward propogation
        %%%%%%%%%%%%%%%%%%%%%%%%
        % 1x1
        Delta3 = -2*(y_trn(n)-X3)*(1-tanh(S3).^2);
        
        % (M2+1)x1 by 1x1 = (M2+1)x1
        Delta2 = W3 * Delta3;
        Delta2 = Delta2(2:end, 1);
        % M2x1 dot M2x1 = M2x1
        Delta2 = Delta2 .* (1-tanh(S2).^2);
        
        % (M1+1)x(M2) by (M2)x1 = (M1+1)x1
        Delta1 = W2 * Delta2;
        Delta1 = Delta1(2:end, 1);
        % M1x1 dot M1x1 = M1x1
        Delta1 = Delta1 .* (1-tanh(S1).^2);
        
        %%%%%%%%%%%%%%%%%%%%%%%%
        %% Gradient decent
        %%%%%%%%%%%%%%%%%%%%%%%%
        % W3: (M2+1)x1, X2*Delta3': (M2+1)x1 by 1x1
        W3 = W3 - eta*X2*Delta3';
        
        % W2: (M1+1)xM2, X1*Delta2': (M1+1)x1 by 1xM2
        W2 = W2 - eta*X1*Delta2';
        
        % W1: (d+1)xM1, X0*Delta11: (d+1)x1 by 1xM1
        W1 = W1 - eta*X0*Delta1';
        
    end
    
    %%%%%%%%%%%%%%%%%%%%%%%%
    %% Test
    %%%%%%%%%%%%%%%%%%%%%%%%
    K = size(X_tst, 1);
    
    S1 = W1' * X_tst';
    X1 = [ones(1, K); tanh(S1)];
    S2 = W2' * X1;
    X2 = [ones(1, K); tanh(S2)];
    S3 = W3' * X2;
    X3 = tanh(S3);
    
    y_pred = sign(X3');
    error = sum(y_pred ~= y_tst) / K;
    
    err_mat(ix) = error;
    fprintf('Done %d, %f\n', ix, error);

end

avg_error = mean(err_mat);
fprintf('avg error: %f\n', avg_error);