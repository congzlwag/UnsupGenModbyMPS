function [C cindex] = tensor_product(varargin)
% Author: Jing Chen yzcj105@126.com

% varargin is cindex
%C(cindex)=A(aindex)*B(bindex)
% the same string in index will be summed up
% A,B,C is muti dimention array

%get all the permute order
if nargin == 4
    A = varargin{1};
    aindex = varargin{2};
    B = varargin{3};
    bindex = varargin{4};
elseif nargin == 5
    cindex = varargin{1};
    A = varargin{2};
    aindex = varargin{3};
    B = varargin{4};
    bindex = varargin{5};
end
a_length = length ( aindex );
b_length = length ( bindex );

size_a = size(A);
size_a(end+1:a_length) = 1;
size_b = size(B);
size_b(end+1:b_length) = 1;

[com_in_a, com_in_b ] = find_common ( aindex, bindex );

if ~all(size_a(com_in_a)==size_b(com_in_b))
    error('The dimention doesnot match!');
end

diff_in_a = 1:a_length;
diff_in_a ( com_in_a ) = [];
diff_in_b = 1:b_length;
diff_in_b ( com_in_b ) = [];
temp_idx = [ aindex(diff_in_a) , bindex(diff_in_b) ];

if nargin ==5
    [ ix1 ix2 ] = find_common ( temp_idx , cindex );
    ix_temp (ix2) = ix1 ;
else
    cindex = temp_idx;
end
c_length = length(cindex);
% mutiply
if any([ com_in_a diff_in_a ] ~= 1:a_length)
    A = permute( A, [ com_in_a diff_in_a ] );
end
if any([ com_in_b diff_in_b ] ~= 1:b_length)
    B = permute( B, [ com_in_b diff_in_b ] );
end

sda = prod(size_a(diff_in_a));
sc = prod(size_a(com_in_a));
sdb = prod(size_b(diff_in_b));

A = reshape(A,[sc,sda,1]);
B = reshape(B,[sc,sdb,1]);

C = A.' * B ;

C = reshape(C,[size_a(diff_in_a),size_b(diff_in_b),1,1]);

if c_length > 1 && nargin == 5 && any(ix_temp ~= 1:c_length)
    C = permute(C,ix_temp);
end

function [com_a, com_b] = find_common ( a, b)
% find the common elements
a = a.';
a_len = length( a );
b_len = length( b );
a = a(:,ones (1,b_len) );
b = b( ones(a_len ,1),:);
%[b a] = meshgrid(b,a);
[ com_a ,com_b ] = find ( a == b );
com_a = com_a.';
com_b = com_b.';




