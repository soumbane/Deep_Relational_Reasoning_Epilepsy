function [ F ] = vec_to_mat( v )
%% convert row vector to upper triangular matrix

%load('X_expressive.mat');

%v = X_expressive(1,1:6728);

%v = [1,2,3,4,5,6,0,0];

n = round((sqrt(8 * numel(v) + 1) - 1) / 2);
M = zeros(n, n);

c = 0;
for i2 = 1:n
  for i1 = 1:i2
    if(i1==i2)
       M(i1,i2) = 0;
    else
       c = c + 1;
       M(i1, i2) = v(c);
    end
  end
end

J = M';

F = J+M;

end

