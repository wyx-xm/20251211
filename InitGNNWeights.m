% InitGNNWeights.m
% 初始化 GNN 权重（与 InitPopulationGNN_v2 对齐）
function w = InitGNNWeights(cfg)
    if nargin < 1, cfg = config(); end
    rng(2);
    % 统一输入维度 (must >= max(targetFeatDim, robotFeatDim))
    inDim = 8; % 这里设为 8 (足够装下 target:4, robot:3 等)
    hidden = max(16, cfg.gnn.hidden);

    w.inDim = inDim;
    w.hidden = hidden;
    % W1: inDim x hidden
    w.W1 = randn(inDim, hidden) * 0.05;
    % W2: hidden x numAUVs (scores for each robot)
    w.W2 = randn(hidden, cfg.numAUVs) * 0.05;
end
