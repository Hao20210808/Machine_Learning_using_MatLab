%% AI & ML HW6
% Programer: PO-HSUN WU
% Last edit date: 2022/10/15 14:00

%% Main function
clear;clc
X = [0 0 1
    0 1 1
    1 0 1
    1 1 1];
D = [0 1 1 0]';

alpha = 0.9;
beta = 0.9;
error_criteria = 1e-3;
epoch_max = 1e4;

W1 = 2*rand(4,3)-1;
W2 = 2*rand(1,4)-1;

epoch = 0;

while 1
    epoch = epoch+1;
    [W1, W2, error] = BackpropMmt(W1, W2, X, D, alpha, beta);
    if ((error <= error_criteria) || (epoch >= epoch_max))
        fprintf("Stop when #%d epochs and error=%.6f.\n", epoch, norm(error))
        break
    end
end

% Test the trained model
N = 4;
y = zeros(4,1);
for k = 1:N
    x = X(k,:)';
    v1 = W1*x;
    y1= Sigmoid(v1);
    v = W2*y1;
    y(k) = Sigmoid(v);
end
disp('y =')
disp(y)

%% Function of momentum method
function [W1, W2, sum_e] = BackpropMmt(W1, W2, X, D, alpha, beta)
    mmt1 = zeros(size(W1));
    mmt2 = zeros(size(W2));
    N = 4;
    sum_e = 0;
    for k = 1:N
        x = X(k,:)';
        d = D(k);
        v1 = W1*x;
        y1 = Sigmoid(v1);
        v = W2*y1;
        y = Sigmoid(v);
        e = d - y;
        delta = y.*(1-y).*e;
        e1 = W2'*delta;
        delta1 = y1.*(1-y1).*e1;
        dW1 = alpha*delta1*x';
        mmt1 = dW1 + beta*mmt1;
        W1 = W1 + mmt1;
        mmt2 = alpha*delta*y1';
        W2 = W2 + mmt2;

        sum_e = sum_e + abs(e);
    end
end

%%
function y = Sigmoid(x)
    y = 1./(1+exp(-x));
end
