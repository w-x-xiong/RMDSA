function x = sub(X,D)
% Subspace algorithm
% --------------------------------
% x = sub(X,r);
% x = 2D position estimate
% X = matrix for receiver positions
% r = TOA measurement vector

%In our case, this function should be invoked as sub(anc,M), in order to 
%output the location estimate


Y = X';
L = size(Y,1); % number of receivers

% [U,Lamda] = eig(D);
[U,S,V] = svd(D);
Un = U(:,3:end);
x = (Y'*(Un*Un')*ones(L,1))/(ones(L,1)'*(Un*Un')*ones(L,1));
