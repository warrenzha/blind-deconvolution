%% 
clear all;  clc
%% Parameters
N = 16;
K = 16;
L = 256;

%% Generate convolution observation y
% Normalize m, h
h = randn(K,1);
h = h/norm(h);
m = randn(N,1);
m = m/norm(m);

% Calculate w = Bh, x = Cm
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

% Convolution
y = real(ifft(fft(x).*fft(w)));

%% Convert to Fourier domain
B_hat = fft(B);
C_hat = fft(C);
y_hat = fft(y);

%% Linear operator
% A = zeros(L,K*N);
% for i=1:size(C_hat,2)
%     A_l = diag(sqrt(L)*C_hat(:,i));
%     A(:,(i-1)*K+1:i*K) = A_l * B_hat;
% end

A = [];
for i= 1:N
    A_l = diag(sqrt(L) * C_hat(:,i));
    A = [A A_l*B_hat];
end

%% CVX optimization
cvx_begin
    variable X(K,N) 
    minimize( norm_nuc(X) )
    subject to
        A*X(:) == y_hat
cvx_end

%% SVD recover h, m from X0
[U,S,V] = svd(X);
u = U(:,1);
v = V(:,1);
error = norm(u*v' - h*m','fro')/norm(h*m','fro')
