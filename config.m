% config.m
% 全局参数配置（统一集中修改）
function cfg = config()
    cfg.numAUVs = 4;
    cfg.numTargets = 20;
    cfg.mapSize = 1000;
    cfg.depot = [100,100];
    cfg.maxGen = 120;
    cfg.simTimeStep = 1.0;
    cfg.timeJump = 50; % simulation time jump per gen for flow dynamics

    % DRL / PPO / NN
    cfg.drl.learningRate = 1e-3;
    cfg.drl.gamma = 0.99;
    cfg.drl.hiddenDim = 64;
    cfg.drl.epsilon = 0.9;
    cfg.drl.epsilonDecay = 0.995;

    cfg.ppo.batchSize = 64;
    cfg.ppo.epochs = 4;
    cfg.ppo.clip = 0.2;
    cfg.ppo.entropyCoef = 0.01;
    cfg.ppo.vfCoef = 0.5;

    % Safe RL
    cfg.safe.initLambdaVec = [0.5, 0.5, 0.5]; % collision, energy, time
    cfg.safe.lambda_lr = 0.05;
    cfg.safe.thresholds = [100, 1e5, 3000]; % sample thresholds

    % communication
    cfg.comm.triggerThresh = 0.03;
    cfg.comm.range = 600;

    % physical
    cfg.phys.mass = 25;
    cfg.phys.volume = 0.03;
    cfg.phys.rho = 1025;
    cfg.phys.Iz = 2.0;
    cfg.phys.maxThrust = 80;
    cfg.phys.maxTorque = 15;
    cfg.phys.batteryCap = 50000;

    % GNN dims
    cfg.gnn.jobFeatDim = 3;
    cfg.gnn.robotFeatDim = 3;
    cfg.gnn.hidden = 16;
    cfg.gnn.edgeDim = 2;

    % Predictor (CNN-LSTM) hyperparams (skeleton)
    cfg.pred.window = 8;
    cfg.pred.cnnFilters = [8,16];
    cfg.pred.lstmHidden = 32;

    % population
    cfg.popSize = 40;

    % visualization
    cfg.visual.updateEvery = max(1, floor(cfg.maxGen/80));
end
