clear;clc;close all

X = [ 0 0 1 ;
      0 1 1 ;
      1 0 1 ;
      1 1 1 ;
      ];

D = [ 0
      0
      1
      1
      ];

E1 = zeros(1000, 1);
E2 = zeros(1000, 1);

W1 = 2*rand(1, 3)-1;
W2 = W1;

for epoch = 1:1000
    W1 = DeltaSGD(W1, X, D);
    W2 = DeltaBatch(W2, X, D);
    es1 = 0;
    es2 = 0;
    N = 4; %inference
    for k = 1:N
        x = X(k, :)';
        d = D(k);
        v1 = W1*x;
        y1 = Sigmoid(v1);
        es1 = es1 + (d - y1)^2;
        v2 = W2*x;
        y2 = Sigmoid(v2);
        es2 =es2 + (d-y2)^2;
    end
    E1(epoch) = es1/N;
    E2(epoch) = es2/N;
end

plot(E1, 'r')
hold on
plot(E2, 'b:')
xlabel('Epoch')
ylabel('Average of Train error')
legend('SGD', 'Batch')


function y = Sigmoid(x)
y = 1 / (1+exp(-x)) ;
end

function W = DeltaBatch(W, X, D) %Main function
    alpha = 0.9;
    dWsum = zeros(3, 1);
    N = 4;
    for k = 1:N

        x = X(k, :)';
        d = D(k);

        v = W*x;
        y = Sigmoid(v);

        e = d - y;
        delta = y*(1-y)*e ;

        dW = alpha*delta*x; %delta rule
        dWsum = dWsum + dW;
    end
    dWavg = dWsum/N;

    W(1) = W(1) + dWavg(1);
    W(2) = W(2) + dWavg(2);
    W(3) = W(3) + dWavg(3);
end

function W = DeltaSGD(W, X, D) %Main function
    alpha = 0.9;
    N = 4; %HW02 change this use matlab function to determine the number of the input data points.
    for k = 1:N

        x = X(k, :)';
        d = D(k);

        v = W*x;
        y = Sigmoid(v);

        e = d - y;
        delta = y*(1-y)*e ;

        dW = alpha*delta*x; %delta rule
        W(1) = W(1) + dW(1);
        W(2) = W(2) + dW(2);
        W(3) = W(3) + dW(3);
    end
end