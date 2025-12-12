function d = CalculateDiversity(individual)
% CalculateDiversity.m
% 计算种群个体的简单多样性指标（与任务分配相关）
% 输入：
%   individual.assign : 1 x numTargets，取值为 1..numAUVs
% 输出：
%   d : 一个标量，代表该个体的多样性（用于 RL state）

    % 如果 individual.assign 不存在或为空 → 返回 0
    if ~isfield(individual, 'assign') || isempty(individual.assign)
        d = 0;
        return;
    end

    a = individual.assign(:);        % 列向量
    % 多样性：使用分配值的标准差（越分散越大）
    d = std(double(a));

    % 如果 std 计算结果为 NaN（例如所有任务未分配），改为 0
    if isnan(d)
        d = 0;
    end
end
