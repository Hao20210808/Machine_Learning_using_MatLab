clear 
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

W = [8 8 8];


%HW01 do not use fix numbers 1:10000, 
%modified the program. if {||W(new) - W(old)|| < E } then stop endif. 
%use flexble weight to run the function
for epoch = 1:10000  %train 
    W = DeltaSGD(W, X, D);
end

N = 4; %inference


for k = 1:N
    x = X(k, :)';
    v = W*x;
    y = Sigmoid(v)
end


function y = Sigmoid(x)
y = 1 / (1+exp(-x)) ;
end


function W = DeltaSGD(W, X, D) %Main function
    alpha = 0.9;

    N = 4; 
%HW02 change this use matlab function to determine the number of the input data points.
%---------------------------------------------------------------------------------------------
%   N = 4;
%OUTPUT=>
%y =0.0021
%y =0.0095
%y =0.9906
%y =0.9979
%---------------------------------------------------------------------------------------------
%   N = 3;
%OUTPUT=>
%y =0.9921
%y =0.0094
%y =0.9905
%y =0.9979
%---------------------------------------------------------------------------------------------
%   N = 2;
%OUTPUT=>
%y =4.3404e-04
%y =0.0066
%y =0.5642
%y =0.9522
%---------------------------------------------------------------------------------------------
%   N = 1;
%OUTPUT=>
%y =0.0093
%y =0.0066
%y =0.5642
%y =0.9522
%---------------------------------------------------------------------------------------------
%   N = 0;
%OUTPUT=>
%y =0.9997
%y =0.9655
%y =0.9655
%y =1.0000
%---------------------------------------------------------------------------------------------
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