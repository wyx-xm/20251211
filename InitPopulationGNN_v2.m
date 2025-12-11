% InitPopulationGNN_v2.m
% 基于 GNN 的初始种群生成（修正版）
% 使用二部图：Targets (1..numT) 与 Robots (numT+1..numT+numA)
% 返回：
%   pop      - population struct array
%   att      - attention matrix (numT x numA)
%   G        - graph struct 包含 .jobEmb, .robEmb, .A, .X, .Score, .att

function [pop, att, G] = InitPopulationGNN_v2(cfg, scene, weights, env)
    % 参数检查
    numT = cfg.numTargets;
    numA = cfg.numAUVs;

    % --- 1) 构建节点特征 ---
    % Target features: [x, y, priority, flowCost]
    tPos = scene.targets;        % numT x 2
    tPri = scene.priorities;     % numT x 1

    flowCost = zeros(numT,1);
    for i = 1:numT
        ix = round(tPos(i,1)/20)+1;
        iy = round(tPos(i,2)/20)+1;
        ix = max(1,min(ix,size(env.U,2)));
        iy = max(1,min(iy,size(env.U,1)));
        u = env.U(iy,ix); v = env.V(iy,ix);
        flowCost(i) = norm([u v]);
    end
    X_t = [tPos, tPri, flowCost];   % numT x 4
    X_t = NormalizeFeat(X_t);

    % Robot features: [init_x, init_y, id]
    if isfield(scene,'auvInit')
        auvPos = scene.auvInit;
    elseif isfield(scene,'auvInitPos')
        auvPos = scene.auvInitPos;
    else
        % fallback around depot
        ang = linspace(0,2*pi,numA+1); ang(end)=[];
        auvPos = cfg.depot + 30*[cos(ang'); sin(ang')];
    end
    rID = (1:numA)';
    X_r = [auvPos, rID]; % numA x 3
    X_r = NormalizeFeat(X_r);

    % --- 2) 构建二部图邻接矩阵 (简单 RBF + full bipartite) ---
    N = numT + numA;
    A = zeros(N,N);
    % T-T adjacency via RBF on targets
    D_tt = pdist2(tPos, tPos);
    A(1:numT,1:numT) = exp(-D_tt.^2 / (2 * 300^2));
    % R-R: identity (can be extended)
    A(numT+1:end, numT+1:end) = eye(numA);
    % T-R and R-T: fully connected bipartite
    A(1:numT, numT+1:end) = ones(numT, numA);
    A(numT+1:end, 1:numT) = ones(numA, numT);

    % --- 3) 合并节点特征到统一矩阵 X (N x inDim) ---
    inDim = weights.inDim;
    X = zeros(N, inDim);
    % place target features into first columns
    X(1:numT,1:size(X_t,2)) = X_t;
    % place robot features into first columns (overlap is ok)
    X(numT+1:end,1:size(X_r,2)) = X_r;

    % --- 4) GNN 前向（两层简化） ---
    % W1: inDim x hidden, W2: hidden x numA
    H1 = max(0, A * X * weights.W1);   % N x hidden
    Score = H1 * weights.W2;           % N x numA

    % job-specific scores (take top part)
    Score_t = Score(1:numT,:);         % numT x numA
    att = softmax(Score_t, 2);         % normalize across A dimension (per target)

    % extract embeddings for state builder
    jobEmb = H1(1:numT, :);
    robEmb = H1(numT+1:end, :);

    % --- 5) 生成初始种群 (使用 attention sampling + 小扰动) ---
    popSize = cfg.popSize;
    pop = repmat(struct('assign',[],'seq',[]), popSize, 1);
    for i = 1:popSize
        assign = zeros(1,numT);
        for t = 1:numT
            p = att(t,:);
            % avoid degenerate zero
            if sum(p)==0
                p = ones(1,numA)/numA;
            else
                p = p ./ sum(p);
            end
            assign(t) = randsample(1:numA,1,true,p);
        end
        % small random mutations
        mask = rand(1,numT) < 0.1;
        if any(mask)
            assign(mask) = randi(numA,1,sum(mask));
        end
        pop(i).assign = assign;
        pop(i).seq = randperm(numT);
    end

    % --- 6) 打包 G 结构 ---
    G.A = A;
    G.X = X;
    G.Score = Score;
    G.att = att;
    G.jobEmb = jobEmb;
    G.robEmb = robEmb;
end

function Xn = NormalizeFeat(X)
    Xn = (X - mean(X,1)) ./ (std(X,[],1) + 1e-8);
end

function Y = softmax(X, dim)
    if nargin<2, dim = 2; end
    mx = max(X,[],dim);
    Xs = X - mx;
    ex = exp(Xs);
    Y = ex ./ (sum(ex,dim) + 1e-12);
end
