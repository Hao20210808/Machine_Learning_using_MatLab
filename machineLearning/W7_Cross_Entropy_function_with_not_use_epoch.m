clear all
X = [ 0 0 1;
      0 1 1;
      1 0 1;
      1 1 1;
    ];
D = [ 0
      1
      1
      0
    ];
W1 = 2*rand(4, 3) - 1;
W2 = 2*rand(1, 4) - 1;
% for epoch = 1:10000
%    [W1, W2] = BackpropCE(W1, W2, X, D);
% end
% -----------------------------------------------------
num = 0;
while num == 0
    [W1, W2] = BackpropCE(W1, W2, X, D);

    if (d - y) < e
        num = num + 1;
    end
end
% -----------------------------------------------------
N = 4; % inference
for k = 1 : N
    x = X(k, :)';
    v1 = W1*x;
    y1 = Sigmoid(v1);
    v = W2*y1;
    y = Sigmoid(v)
end

function [W1, W2] = BackpropCE(W1, W2, X, D)
    alpha = 0.9;
    N = 4;
    for k = 1:N
        x = X(k, :)'; %x = a column vector
        d = D(k);
        v1 = W1*x;
        y1 = Sigmoid(v1);
        v = W2*y1;
        y = Sigmoid(v);
        e =  d - y;
        delta = e; %importment part
        e1 = W2'*delta;
        delta1 = y1.*(1-y1).*e1;
        dW1 = alpha*delta1*x';
        W1 = W1 + dW1;
        dW2 = alpha*delta*y1';
        W2 = W2 + dW2;
    end
end
function y = Sigmoid(x)
    y = 1 ./ (1 + exp(-x));
end