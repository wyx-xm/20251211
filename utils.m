% utils.m
% 工具函数合集
function y = softmax(x, dim)
    if nargin<2, dim = 2; end
    ex = exp(x - max(x,[],dim));
    y = ex ./ sum(ex,dim);
end

function th = angdiff(t1,t2)
    th = angle(exp(1i*(t2 - t1)));
end
