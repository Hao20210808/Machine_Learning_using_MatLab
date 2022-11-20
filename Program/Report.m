%% Main function
clear;clc
X = [-5 -5
     -5  5
      5 -5
      5  5];
D = [-5 5 5 -5]';

alpha = 0.01;
beta = 0.9;
error_criteria = 1e-3;
epoch_max = 1e4;

W1 = rand(2,2);
W2 = rand(1,2);

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
    y1= Comparator(v1);
    v = W2*y1;
    y(k) = Comparator(v);
end
disp('y =')
disp(y)

%%
clear;clc;close all
x = -5:0.001:5;

y = zeros(size(x));
for i = 1:length(x)
    y(i) = Comparator(x(i));
end

figure; hold on
plot(x, Comparator_tanh(x))
plot(x, y)

figure
plot(x, abs(Comparator_tanh(x)-y))

figure
plot(x, Comparator_tanh_derivative(x))

%% Function of momentum method
function [W1, W2, avg_e] = BackpropMmt(W1, W2, X, D, alpha, beta)
    mmt1 = zeros(size(W1));
    mmt2 = zeros(size(W2));
    N = 4;
    sum_e = 0;
    
    for k = 1:N
        x = X(k,:)';
        d = D(k);

        v1 = W1*x;
        y1 = Comparator_tanh(v1);

        v2 = W2*y1;
        y2 = Comparator_tanh(v2);

        e = d - y2;
        e1 = e;
        delta2 = Comparator_tanh_derivative(v2).*e1;
        dW2 = alpha*delta2*y2';

        e2 = W2'*delta2;
        delta1 = Comparator_tanh_derivative(v1).*e2;
        dW1 = alpha*delta1*y1';

        mmt1 = dW1 + beta*mmt1;
        W1 = W1 + mmt1;
        mmt2 = dW2 + beta*mmt2;
        W2 = W2 + mmt2;

        sum_e = sum_e + abs(e);
    end
    avg_e = sum_e/N;
end

%%
function Vout = Comparator(Vp)
    Vcc = +5;
    Vee = -5;
    Vn = 5*(22/122);
    if Vp>Vn
        Vout = Vcc;
    elseif Vp<Vn
        Vout = Vee;
    else
        Vout = 0;
    end
end

%%
function y = Comparator_tanh(x)
    V = 5;
    Vn = 5*(22/122);
    A = 10;
    y = V*tanh(A*(x-Vn));
end

function y = Comparator_tanh_derivative(x)
    V = 5;
    Vn = 5*(22/122);
    A = 10;
    y = V*A*(1-tanh(A*(x-Vn)).^2);
end
