% Use MATLAB functions to draw surface of the cost function given in p.1071, 
% phi(alpha2,alpha3). 
% Indicates the optimal (alpha2, alpha3) and its associate function values, 
% also try to explain why the optimal alpha3 can be negative in this case.
clc; clear; close all;
%% cost function with phi(alpha2,alpha3)

%parameter x and Class t (three samples of data)
x=[2 2; 4 5; 7 4];
t=[-1; 1; 1];
N=length(t); 
%N=3

K = x*x';
h = (t*t').*K + 1e-5*eye(N);

f = repmat(1,N,1); 
% array with 1 (col=N, row=1)

A = []; 
% Matrix of inequality constraints, no inequality constraints

b = []; 
% right hand side of the inequality constraints, no inequality constraints

LB = repmat(0,N,1); 
% array with 0 (col=N, row=1)
% lb ≤ x : lower bounds for x, no lower bounds

UB = repmat(inf,N,1); 
% array with Infinity (col=N, row=1)
% x ≤ ub : upper bounds for x, no upper bounds

Aeq = t';
beq = 0;

% bias
alpha = quadprog(h,-f, A, b, Aeq,beq,LB,UB);
fout = sum(repmat(alpha.*t,1,N).*K,1)';
% optimal value of the objective function

pos = find(alpha>1e-6);
% optimal solution

bias = mean(t(pos)-fout(pos));
% CostFunction
Error=(h-t).^2;
J=(1/(2*N))*sum(Error);

fprintf("J = %d\n", J);
fprintf("----------------\n");
disp(J);

for i=1:N
    Error=h-t;
    delta=x'*Error;
    t=t-(alpha/N)*sum(sum(delta));
    h=t.*x;
    endError=(h-t).^2;
    J=(1/(2*N))*sum(Error);
    disp(J);
end

% plot
