%% 
clear all; clc
%% load image

rgb = double(imread('shapes3.png'));
x = mean(rgb,3);
x = double(x)/norm(x,'fro');
imagesc(x); title('Original shapes image'),colormap(gray);
L1 = size(x,1); 
L2 = size(x,2); 
L = L1*L2;

%% blur kernel

blur_kernel = fspecial('motion',5,45);
% h1 = fspecial('disk',7);
% h1 = fspecial('gaussian',[20,20],5)
[K1 K2] = size(blur_kernel); 
blur_kernel = blur_kernel/norm(blur_kernel,'fro');
w = zeros(L1,L2);
w(L1/2-(K1+1)/2+2:L1/2+(K1+1)/2,L2/2-(K2+1)/2+2:L2/2+(K2+1)/2) = blur_kernel; % K1 and K2 are odd; change if K1 and K2 even

%% Useful functions

mat = @(x) reshape(x,L1,L2);
vec = @(x) x(:);
%% Computing matrix B; see blind deconvolution using convex programming paper for notations 

w_vec = vec(w);
Indw = zeros(L,1);
Indw = abs(w_vec)>0;
figure;
plot(Indw);

j = 1;
K = sum(Indw);
B = sparse(L,K); 
h = zeros(K,1);
for i = 1:L
    if(Indw(i) == 1)
        B(i,j) = Indw(i);
        h(j) =w_vec(i);
        j = j+1;
    end
end

%% Define function BB

BB = @(x) mat(B*x);
BBT = @(x) B'*vec(x);
w1 = BB(h);   % w = Bh
figure;
imagesc(mat(w1)),title('blur kernel'), colormap(gray), colorbar;

%% 2D convolution

figure;
conv_wx = ifft2(fft2(x).*fft2(BB(h)));
conv_wx_image = fftshift(mat(conv_wx));
figure;
imagesc(conv_wx_image),title('Convolution of original image with blur kernel'), colormap(gray);
%% Compute and display wavelet coefficients of the original and blurred image

[alpha_conv,l] = wavedec2(conv_wx_image,4,'db1');
figure;
plot(alpha_conv); title('wavelet coefficients of the convolved image');

[alpha_x,l] = wavedec2(x,4,'db1');
figure;
plot(alpha_x); title('wavelet coefficients of the original image');

%% C selected by wavelet coeffs of blurred\original\both image

alpha = alpha_x;
Ind = zeros(1,length(alpha));
Ind_alpha_conv = abs(alpha_conv)>0.00018*max(abs(alpha_conv)); 
% Ind_alpha_conv is support recovered from blurred image; For actual recovery without oracle
% info
Ind_alpha_x = abs(alpha_x)>0.0005*max(abs(alpha_x)); 
% Ind_alpha_x is support recovered from original image; For oracle assisted
% recovery 

% Ind_alpha_x = zeros(1,length(alpha)); % Do this if you want to kill support info. from original image
Ind_alpha_conv = zeros(1,length(alpha)); % Do this if you want to kill support info. from blurred image
Ind = ((Ind_alpha_conv>0)|(Ind_alpha_x>0)); % Taking union of both supports

fprintf('Number of non-zeros in x estimated from the blurred image: %.3d\n', sum(Ind_alpha_conv));
fprintf('Number of non-zeros in x estimated from the original image: %.3d\n', sum(Ind_alpha_x));
fprintf('Union of the non-zero support from original and blurred image: %.3d\n', sum(Ind));

figure;
plot(Ind);

%% Compute matrix C; see blind deconvolution paper for notations

j = 1;
N = sum(Ind);
C = sparse(L,N);
for i = 1:size(alpha,2)
    if(Ind(i) == 1)
        C(i,j) = Ind(i);
        m(j) = alpha(i);
        j = j+1;
    end
end
m = m';

%% Define function CC

[c,l] = wavedec2(conv_wx_image,4,'db1');
CC = @(x) waverec2(C*x,l,'db1');
CCT = @(x) (C'*(wavedec2(x,4,'db1'))');

%% Approximated convolved image

x_hat = waverec2(C*m,l,'db1');
fprintf('Origonal image vs Wavelet approximated image: %.3e\n', norm(x-x_hat,'fro')/norm(x,'fro'));
figure;
imagesc(x_hat), title('Approximation of original image from few coeffs'), colormap(gray), colorbar; 

%% Blind deconvolution using convex programming

kernel = blur_kernel(:);
K = sum(kernel~=0);
B = zeros(length(kernel),K);
idx = 1;
h = zeros(K,1);
for i = 1:length(kernel)
    if kernel(i) ~= 0
        B(i,idx) = 1;
        h(idx) = kernel(i);
        idx = idx + 1;            
    end
end

[C_haar,s] =  wavedec2(x, 4,'db1'); 
N = sum(C_haar~= 0);
img = x(:);
C = zeros(length(img),N);
m = zeros(N,1); 
idx = 1;
for i=1:length(C_haar)
    if C_haar(i) ~= 0
        C(i,idx) = 1;
        m(idx) = C_haar(i);
        idx=idx +1;
    end
end

y = conv_wx_image(:);
L = length(y);
y_hat = dftmtx(L)*y;
B_hat = dftmtx(L)*B;
C_hat = dftmtx(L)*C;

A = [];
for i= 1:N
    A_l = diag(sqrt(L) * C_hat(:,i));
    A = [A A_l*B_hat];
end

cvx_begin
    variable X(K,N) 
    minimize( norm_nuc(X) )
    subject to
        A*X(:) == y_hat
cvx_end

[U,S,V] = svd(X);
u = U(:,1);
v = V(:,1);

C_recover = C*v.*(-1);
x_dec = waverec2(C_recover, s, 'haar');
imagesc(x_dec); title('Recovered image using convex programming'),colormap(gray);

