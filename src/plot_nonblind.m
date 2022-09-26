%% 
clear all;  clc
%% Blind program
global L B A K N y;

P_blind = zeros(11,1);
error_convex = zeros(11,50);

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
        B = B(:,idxB);
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
        
        error_convex(col,row) = error;
        if error<0.02
            P_blind(col) = P_blind(col) + 0.02;
        end
    end
end

%% Non-blind program
P_success_nonblind = zeros(11,1);
Error_nonblind = zeros(11,50);
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
        B = B(:,idxB);
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

        z0 = [randn(K,1);zeros(L-K,1)];
        fun = @(z) norm(real(ifft(fft(x).*fft(z))) - y);
        [z,fval] = fmincon(fun,z0);
        error = norm(z*x' - w*x','fro')/norm(w*x','fro');
        if error<0.02
            P_success_nonblind(col) = P_success_nonblind(col) + 0.02;
        end
    end
end

%% Plot comparison between blind and non-blind -- Fig. 2
plot(linspace(1,11,11),P_success_nonblind,'-x',linspace(1,11,11),P_blind,'-o');
xlabel('L/(K+N)');
ylabel('Success rate');
xlim([1,11]);
ylim([-0.05, 1.05]);
xticks(linspace(1,11,5));
set(gca,'xticklabel',{'1','1.5','2','2.5','3'});
legend('non-blind','blind','Location','southeast');
title('Transition curve (non-blind, blind)');
set(gca,'FontSize',12);
grid on;
