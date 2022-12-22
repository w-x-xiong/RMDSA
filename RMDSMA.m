function [M,R,C,fail] = RMDSMA(D_hat,rho,r,maxiter)
%Robust Multidimensional Similarity Matrix Approximation Algorithm
%by Alternating Direction Method of Multipliers
%with optimization variables U, V, R, C
%Note that the low-rank structure is guaranteed by the product M=UV

%The optimization problem is of the form
%   argmin_(U,V,R,C) ||R||_2,1 + ||C'||_2,1
%   s.t.   UV+R+C = D_hat

%-Inputs
%D_hat - row-column structured noise corrupted matrix
%r - target rank of M 
%rho - penalty parameter in augmented Lagrangian function
%maxiter - max iteration number

%-Outputs
%M - approximated low-rank matrix
%R - row-sparse outlier matrix
%C - column-sparse outlier matrix
%fail - true if fail to converge, otherwise false


sz = size(D_hat);
%initialization
M = D_hat;
R = zeros(sz);
C = zeros(sz);
Lmbd = zeros(sz);
fail = false;
%k: counter for iterations
k = 0;
while 1
    k = k + 1;
    if k > maxiter
        fprintf('It cannot reach the convergence criterion in %d iterations\n', maxiter)
        fail = true;
        break
    end
    %ADMM section
    %update U,V
    N = -(R+C-D_hat+Lmbd/rho);
    [U_t,S_t,V_t] = svds(N,r);
    U_new = U_t*(S_t^(1/2));
    V_new = (S_t^(1/2))*V_t';
    
    %update R
    T = -((Lmbd/rho)+U_new*V_new+C-D_hat);
    R_new = R;
    for j = 1:length(R_new)
        R_new(j,:) = max(0,1-(1/(rho*norm(T(j,:)))))*T(j,:);
    end
    
    %update C
    T2 = -((Lmbd'/rho)+(U_new*V_new)'+R_new'-D_hat');
    C_new = C;
    for j = 1:length(C_new)
        C_new(j,:) = max(0,1-(1/(rho*norm(T2(j,:)))))*T2(j,:);
    end
    C_new = C_new';
    
    %update Lmbd
    Lmbd = Lmbd+rho*(U_new*V_new+R_new+C_new-D_hat);
    
    R = R_new;
    C = C_new;
    
    if (norm(U_new*V_new+R_new+C_new-D_hat, 'fro')/norm(D_hat,'fro'))<1e-7
        M = U_new*V_new;
        break
    end
    
end
fprintf('It takes %d iterations due to ADMM\n', k)
end

