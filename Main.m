% Main.m
% 主程序入口：初始化、训练/仿真循环、启动 GUI 可视化
% 直接运行：在 MATLAB 命令窗口输入 `Main` 并回车

function Main()
    close all; clear; clc;
    fprintf('=== NextGen DeepHRL AUV (Full) - Starting ===\n');

    % 载入配置
    cfg = config();

    % 生成初始场景与涡流种子
    scene = GenerateScenario(cfg);
    envSeed = InitVortexEnvironment(cfg.mapSize);

    % 初始化模块：GNN 权重、Predictor、Actor-Critic、Replay
    gnnWeights = InitGNNWeights(cfg);
    predictor = Predictor_init(cfg); % predictor 包含 CNN-LSTM 的参数占位
    AC = ActorCritic_init(cfg);      % Actor-Critic (PPO 简化)
    buffer = ReplayBuffer(cfg);

    % 初始种群（GNN 引导）
    [pop, attMatrix, G] = InitPopulationGNN_v2(cfg, scene, gnnWeights, GetVortexEnv(envSeed,0,cfg.mapSize));

    % 可视化 GUI（非阻塞）
    gui = VisualizeGUI(cfg, scene); % 返回 GUI 句柄（后续可更新）

    % 训练/优化循环（主循环）
    globalBestCost = inf;
    globalBestSol = [];
    hist.cost = []; hist.safety = []; hist.comm = 0;
    lambda_vec = cfg.safe.initLambdaVec; % vector for multiple constraints

    currentTime = 0;
    for gen = 1:cfg.maxGen
        % 更新流场和场景短时状态
        env = GetVortexEnv(envSeed, currentTime, cfg.mapSize);

        % GNN 前向（生成 embeddings，用于 Actor/Critic）
        [pop, attMatrix, G] = InitPopulationGNN_v2(cfg, scene, gnnWeights, env);

        % 评估（估计成本，部分精算）
        [rawCosts, safetyVio, energyCons, paths] = EvaluateSwarm(pop, cfg, env, scene, predictor);

        % Safe RL 惩罚向量化（collision, energy, time）
        penalties = max(0, safetyVio - cfg.safe.thresholds(:)'); % row vector
        % total penalized fitness per individual
        fitness = rawCosts + penalties * lambda_vec';

        % 更新拉格朗日乘子（向量）
        avgPen = mean(max(0, safetyVio - cfg.safe.thresholds(:)'),1);
        lambda_vec = max(0, lambda_vec + cfg.safe.lambda_lr * avgPen);

        % 选择当前最优
        [currMin, idx] = min(fitness);
        if currMin < globalBestCost * (1 - cfg.comm.triggerThresh)
            globalBestCost = currMin;
            globalBestSol = pop(idx);
            hist.comm = hist.comm + 1;
        end

        % Actor-Critic 策略采样 & store experience
        % build states for each candidate job (using G embeddings)
        % Here we collect trajectories per individual -> store into buffer
        for i = 1:length(pop)
            % state representation: global G readout + pop-specific features
            s = buildStateFromG(G, pop(i), cfg);
            [action, logp, v] = ActorCritic_policy(AC, s);
            % We evaluate after action by re-estimating cost (one-step)
            % Use predicted outcome as reward proxy
            r = -fitness(i); % use negative cost as reward
            done = false;
            buffer = buffer.push(s, action, r, done, logp, v);
        end

        % Periodically update Actor-Critic with PPO-like update
        if buffer.size() >= cfg.ppo.batchSize
            batch = buffer.sample(cfg.ppo.batchSize);
            AC = PPO_update(AC, batch, cfg);
            buffer.clear(); % simple schedule
        end

        % Evolutionary Operator (auction + flow aware)
        pop = EvolveWithAuctionAndFlow(pop, fitness, attMatrix, cfg, env, scene, gnnWeights);

        % Logging
        hist.cost(end+1) = globalBestCost;
        hist.safety(end+1) = mean(safetyVio(:));
        fprintf('Gen %d | BestCost %.2f | AvgSafety %.2f | Comm %d\n', gen, globalBestCost, mean(safetyVio(:)), hist.comm);

        % Update GUI every few generations
        if mod(gen, max(1,floor(cfg.maxGen/50)))==0
            gui.update(env, scene, globalBestSol, hist);
        end

        % advance time and decay epsilon
        currentTime = currentTime + cfg.simTimeStep * cfg.timeJump;
        cfg.drl.epsilon = cfg.drl.epsilon * cfg.drl.epsilonDecay;
    end

    % Final visualization
    gui.finalize();
    save('trained_AC.mat','AC','predictor','gnnWeights','cfg','scene');
    fprintf('=== Run Complete. Results saved to trained_AC.mat ===\n');
end

function s = buildStateFromG(G, individual, cfg)
    % 简化的 state builder：全局 node mean + individual's diversity metrics
    jobEmb = mean(G.jobEmb,1);
    robEmb = mean(G.robEmb,1);
    diversity = CalculateDiversity(individual);
    s = [jobEmb, robEmb, diversity];
end

