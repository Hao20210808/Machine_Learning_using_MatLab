%% Main
clear all

load('W11_MnistConv.mat')
k  = 2;
x  = X(:, :, k);
y1 = Conv(x, W1);
y2 = ReLU(y1);
y3 = Pool(y2);
y4 = reshape(y3, [], 1);
v5 = W5*y4;
y5 = ReLU(v5);
v  = Wo*y5;
y  = Softmax(v);

figure;
display_network(x(:));
title('Input Image')

convFilters = zeros(9*9, 20);
for i = 1:20
    filter = W1(:, :, i);
    convFilters(:, i) = filter(:);
end

figure
display_network(convFilters);
title('Convolution Filters')

fList = zeros(20*20, 20);
for i = 1:20
    feature = y1(:, :, i);
    fList(:, i) = feature(:);
end

figure
display_network(fList);
title('Features [Convolution]')

fList = zeros(20*20, 20);
for i = 1:20
    feature = y2(:, :, i);
    fList(:, i) = feature(:);
end

figure
display_network(fList);
title('Features[Convolution + ReLU]')

fList = zeros(10*10, 20);
for i = 1:20
    feature = y3(:, :, i);
    fList(:, i) = feature(:);
end

figure
display_network(fList);
title('Features[Convolution + ReLU + MeanPool]')

%% Conv
function y = Conv(x, W)

    [wrow, wcol, numFilters] = size(W);
    [xrow, xcol,~] = size(x);
    
    yrow = xrow - wrow +1;
    ycol = xcol - wcol +1;
    
    y = zeros(yrow, ycol, numFilters);

    for k = 1:numFilters
        filter = W(:, :, k);
        filter = rot90(squeeze(filter), 2);
        y(:, :, k) = conv2(x, filter,'valid');
    end
end

%% Pooling
function y = Pool(x)
    [xrow, xcol, numFilters]=size(x);
    y = zeros(xrow/2, xcol/2, numFilters);
    filter = ones(2) / (2*2); % for mean

    for k = 1:numFilters
        image = conv2(x(:, :, k), filter,'valid');
        y(:, :, k) = image(1:2:end, 1:2:end);
    end
end

%% ReLU
function y = ReLU(x)
    y = max(0, x);
end

%% Softmax
function y = Softmax(x)
    ex = exp(x);
    y = ex / sum(ex);
end

%% display_network
function [h, array] = display_network(A, opt_normalize, opt_graycolor, cols, opt_colmajor)
    warning off all

    if ~exist('opt_normalize', 'var') || isempty(opt_normalize)
        opt_normalize= true;
    end

    if ~exist('opt_graycolor', 'var') || isempty(opt_graycolor)
        opt_graycolor= true;
    end

    if ~exist('opt_colmajor', 'var') || isempty(opt_colmajor)
        opt_colmajor = false;
    end

% rescale
    A = A - mean(A(:));

    if opt_graycolor, colormap(gray); end

% compute rows, cols
    [L M]=size(A);
    sz=sqrt(L);
    buf=1;
    if ~exist('cols', 'var')
        if floor(sqrt(M))^2 ~= M
            n=ceil(sqrt(M));
            while mod(M, n)~=0 && n<1.2*sqrt(M), n=n+1; end
            m=ceil(M/n);
        else
            n=sqrt(M);
            m=n;
        end
    else
        n = cols;
        m = ceil(M/n);
    end

    array=-ones(buf+m*(sz+buf),buf+n*(sz+buf));

    if ~opt_graycolor
        array = 0.1.* array;
    end


    if ~opt_colmajor
        k=1;
        for i=1:m
            for j=1:n
                if k>M, 
                    continue; 
                end
                clim=max(abs(A(:,k)));
                if opt_normalize
                    array(buf+(i-1)*(sz+buf)+(1:sz),buf+(j-1)*(sz+buf)+(1:sz))=reshape(A(:,k),sz,sz)/clim;
                else
                    array(buf+(i-1)*(sz+buf)+(1:sz),buf+(j-1)*(sz+buf)+(1:sz))=reshape(A(:,k),sz,sz)/max(abs(A(:)));
                end
                k=k+1;
            end
        end
    else
        k=1;
        for j=1:n
            for i=1:m
                if k>M, 
                    continue; 
                end
                clim=max(abs(A(:,k)));
                if opt_normalize
                    array(buf+(i-1)*(sz+buf)+(1:sz),buf+(j-1)*(sz+buf)+(1:sz))=reshape(A(:,k),sz,sz)/clim;
                else
                    array(buf+(i-1)*(sz+buf)+(1:sz),buf+(j-1)*(sz+buf)+(1:sz))=reshape(A(:,k),sz,sz);
                end
                k=k+1;
            end
        end
    end

    if opt_graycolor
        h=imagesc(array,'EraseMode','none',[-1 1]);
    else
        h=imagesc(array,'EraseMode','none',[-1 1]);
    end
    axis image off

    drawnow;

    warning on all
end