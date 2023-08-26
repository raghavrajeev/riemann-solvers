AA1u_sparse = load AA1u.dat
AA1u = spconvert(AA1u_sparse)
AA1u_ichol = transpose(ichol(AA1u_sparse))


% try chol(H)
%     disp('Matrix is symmetric positive definite.')
% catch ME
%     disp('Matrix is not symmetric positive definite')
% end

% tf = issymmetric(A)
% disp(tf)

% k=rank(A)
% disp('Rank = ',k)

% for i=m
%     for
%     a = a + i;
% end
