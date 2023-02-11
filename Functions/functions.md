### Sigmoid

    function y = Sigmoid(x)
        y = 1 ./ (1 + exp(-x));
    end


### Dropout

    function ym = Dropout(y, ratio)
        [m, n] = size(y);
        ym = zeros(m, n);

        num = round(m*n*(1 - ratio)); %number of survivors
        idx = randperm(m*n, num);%Random permutation.
        ym(idx) = 1 / (1-ratio);
    end


### ReLU

    function y = ReLU(x)
        y = max(0, x);
    end


### Softmax

    function y = Softmax(x)
        ex = exp(x);
        y = ex / sum(ex);
    end


### Pooling

    function y = Pool(x)
        [xrow, xcol, numFilters]=size(x);
        y = zeros(xrow/2, xcol/2, numFilters);
        filter = ones(2) / (2*2); % for mean

        for k = 1:numFilters
            image = conv2(x(:, :, k), filter,'valid');
            y(:, :, k) = image(1:2:end, 1:2:end);
        end
    end


### Convolution

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


### display_network

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


### Load_Images

    function images = loadMNISTImages(filename)
        %loadMNISTImages returns a 28x28x[number of MNIST images] matrix containing
        %the raw MNIST images

        fp = fopen(filename, 'rb');
        assert(fp ~= -1, ['Could not open ', filename, '']);

        magic = fread(fp, 1, 'int32', 0, 'ieee-be');
        assert(magic == 2051, ['Bad magic number in ', filename, '']);

        numImages = fread(fp, 1, 'int32', 0, 'ieee-be');
        numRows = fread(fp, 1, 'int32', 0, 'ieee-be');
        numCols = fread(fp, 1, 'int32', 0, 'ieee-be');

        images = fread(fp, inf, 'unsigned char');
        images = reshape(images, numCols, numRows, numImages);
        images = permute(images,[2 1 3]);

        fclose(fp);

        % Reshape to #pixels x #examples
        images = reshape(images, size(images, 1) * size(images, 2), size(images, 3));
        % Convert to double and rescale to [0,1]
        images = double(images) / 255;
    end


### Load_Labels

    function labels = loadMNISTLabels(filename)
    %loadMNISTLabels returns a [number of MNIST images]x1 matrix containing
    %the labels for the MNIST images

        fp = fopen(filename, 'rb');
        assert(fp ~= -1, ['Could not open ', filename, '']);

        magic = fread(fp, 1, 'int32', 0, 'ieee-be');
        assert(magic == 2049, ['Bad magic number in ', filename, '']);

        numLabels = fread(fp, 1, 'int32', 0, 'ieee-be');

        labels = fread(fp, inf, 'unsigned char');

        assert(size(labels,1) == numLabels, 'Mismatch in label count');

        fclose(fp);
    end
