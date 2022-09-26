%% 
clear all;  clc
%% 

table = zeros(25,25);
L = 256;

for k = 1:25
    for j = 1:25
        
        N = 5 * k;
        K = 5 * j;
        
        h = randn(K,1);
        h = h/norm(h);
        m = randn(N,1);
        m = m/norm(m);

%         idxB = randperm(L);
%         idxB = idxB(1:K);
%         B = eye(L);
%         B = B(:,idxB);
%         w = B * h;
% 
%         idxC = randperm(L);
%         idxC = idxC(1:N);
%         C = eye(L);
%         C = C(:,idxC);
%         x = C * m;       

        idxB = randperm(L);
        idxB = idxB(1:K);
        B = randn(L,L);
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
        table(k,j) = error;
    end
end

%% plot phase transition -- Fig. 1
imagesc(1-table);
colormap(gray);
set(gca,'YDir','normal');
set(gca,'xticklabel',{'25','50','75','100','125'});
set(gca,'yticklabel',{'25','50','75','100','125'});
title('L = 256');
xlabel('K');
ylabel('N');
set(gca,'FontSize',12)
grid on;
grid minor;
colorbar('Fontsize',11);

