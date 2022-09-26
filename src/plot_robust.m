%% Dense B only
P_denseB = zeros(11,1);
error_denseB = zeros(11,50);

N = 16;
K = 16;

for col = 1:11
    for row = 1:50
        rng(row);
        L = round((0.2 * col + 0.8) * (K + N));
        
        h = randn(K,1);
        h = h/norm(h);
        m = randn(N,1);
        m = m/norm(m);

        idxB = randperm(L);
        idxB = idxB(1:K);
        B = randn(L,L);
        B = B(:,1:K);
        w = B * h;

        idxC = randperm(L);
        idxC = idxC(1:N);
        C = eye(L);
        C = C(:,idxC);
        x = C * m;

        y = real(ifft(fft(x).*fft(w)));
        B_hat = fft(B);
        C_hat = fft(C);
        y_hat = fft(y);

        A = [];
        for i= 1:N
            A_l = diag(sqrt(L) * C_hat(:,i));
            A = [A A_l*B_hat];
        end

        cvx_begin
            variable X(K,N) 
            minimize( norm_nuc(X) )
            subject to
                A*X(:) == y_hat;
        cvx_end

        [U,S,V] = svd(X);
        u = U(:,1);
        v = V(:,1);
        error = norm(u*v' - h*m','fro')/norm(h*m','fro');
        error_denseB(col,row) = error;
        if error<0.02
            P_denseB(col) = P_denseB(col) + 0.02;
        end
    end
end
%% Dense C only
P_denseC = zeros(11,1);
error_denseC = zeros(11,50);

N = 16;
K = 16;

for col = 1:11
    for row = 1:50
        rng(row);
        L = round((0.2 * col + 0.8) * (K + N));
       
        h = randn(K,1);
        h = h/norm(h);
        m = randn(N,1);
        m = m/norm(m);

        idxB = randperm(L);
        idxB = idxB(1:K);
        B = eye(L);
        B = B(:,1:K);
        w = B * h;

        idxC = randperm(L);
        idxC = idxC(1:N);
        C = randn(L);
        C = C(:,idxC);
        x = C * m;

        y = real(ifft(fft(x).*fft(w)));
        B_hat = fft(B);
        C_hat = fft(C);
        y_hat = fft(y);

        A = [];
        for i= 1:N
            A_l = diag(sqrt(L) * C_hat(:,i));
            A = [A A_l*B_hat];
        end

        cvx_begin
            variable X(K,N) 
            minimize( norm_nuc(X) )
            subject to
                A*X(:) == y_hat;
        cvx_end

        [U,S,V] = svd(X);
        u = U(:,1);
        v = V(:,1);
        error = norm(u*v' - h*m','fro')/norm(h*m','fro');
        error_denseC(col,row) = error;
        if error<0.02
            P_denseC(col) = P_denseC(col) + 0.02;
        end
    end
end
%% Dense B and C
P_denseBC = zeros(11,1);
error_denseBC = zeros(11,50);

N = 16;
K = 16;

for col = 1:11
    for row = 1:50
        rng(row);
        L = round((0.2 * col + 0.8) * (K + N));
        
        h = randn(K,1);
        h = h/norm(h);
        m = randn(N,1);
        m = m/norm(m);

        idxB = randperm(L);
        idxB = idxB(1:K);
        B = randn(L,L);
        B = B(:,1:K); %short
        w = B * h;

        idxC = randperm(L);
        idxC = idxC(1:N);
        C = randn(L);
        C = C(:,idxC);
        x = C * m;

        y = real(ifft(fft(x).*fft(w)));
        B_hat = fft(B);
        C_hat = fft(C);
        y_hat = fft(y);

        A = [];
        for i= 1:N
            A_l = diag(sqrt(L) * C_hat(:,i));
            A = [A A_l*B_hat];
        end

        cvx_begin
            variable X(K,N) 
            minimize( norm_nuc(X) )
            subject to
                A*X(:) == y_hat;
        cvx_end

        [U,S,V] = svd(X);
        u = U(:,1);
        v = V(:,1);
        error = norm(u*v' - h*m','fro')/norm(h*m','fro');
        error_denseBC(col,row) = error;
        if error<0.02
            P_denseBC(col) = P_denseBC(col) + 0.02;
        end
    end
end

%% Plot Fig.4
plot(linspace(1,11,11),P_denseBC,'-s',linspace(1,11,11),P_denseB,'-^',linspace(1,11,11),P_success_convex,'-o',linspace(1,11,11),P_denseC,'-*');
xlabel('L/(K+N)');
ylabel('Success rate');
xlim([1,11]);
ylim([-0.05, 1.05]);
xticks(linspace(1,11,5));
set(gca,'xticklabel',{'1','1.5','2','2.5','3'});
legend('Dense B and C','Dense B and sparse C','Sparse B and C','Sparse B and dense C','Location','southeast');
title('Robustness against sparsity');
set(gca,'FontSize',12);
grid on;