%%%%%%%%%%%%%%%%%
%% Initilization
%%%%%%%%%%%%%%%%%
rng(20121228);

data_trn = load('hw4_kmeans_train.dat');
X_trn = data_trn(:, 1:end-1);
y_trn = data_trn(:, end);

[N, d] = size(X_trn);

K = 10;
IX = 500;
err_mat = zeros(IX, 1);

for ix = 1 : IX
    
    Mu = randi(N, K, 1);
    
    Mu = X_trn(Mu, :);
    
    dist_mat = zeros(N, K);
    S_IX_old = zeros(N, 1);
    
    obj = 10000;
    iter = 1;
    while (iter > 0)
        
        for k = 1:K
            x = Mu(k, :);
            x_rep = repmat(x, N, 1);
            x_dist = x_rep - X_trn;
            % N by 1
            x_dist = sum(x_dist .* x_dist, 2);
            dist_mat(:, k) = x_dist;
        end
        
        %% Find the set S1 ... SK based on distance
        [C, S_IX] = min(dist_mat, [], 2);
        
        %% Break condition
        if S_IX_old == S_IX
            break;
        end
        
        S_IX_old = S_IX;
                
        %% Recompute Mu
        for k = 1:K
            Mu(k, :) = mean(X_trn((S_IX == k), :));
        end
        
        iter = iter + 1;
        obj = sum(C);
        fprintf('iter: %d, obj: %f\n', iter, obj);
        
    end
    
    %%%%%%%%%%%%%%%%%
    %% Ein
    %%%%%%%%%%%%%%%%%
    fprintf('IX: %d, iter: %d, obj: %f\n', ix, iter, obj);
    err_mat(ix) = obj / N;

end

avg_error = mean(err_mat);
fprintf('avg error: %f\n', avg_error);