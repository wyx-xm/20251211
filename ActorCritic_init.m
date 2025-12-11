% ActorCritic_init.m
% 初始化 Actor-Critic 网络（PPO 简化实现）
function AC = ActorCritic_init(cfg)
    rng(4);
    sDim = cfg.gnn.hidden * 2 + 1; % approximate state dim (jobEmb + robEmb + diversity)
    aDim = cfg.numAUVs;            % discrete actions per job in this simplified layout
    AC.actor.W1 = randn(sDim, cfg.drl.hiddenDim)*0.01;
    AC.actor.b1 = zeros(1,cfg.drl.hiddenDim);
    AC.actor.W2 = randn(cfg.drl.hiddenDim, aDim)*0.01;
    AC.actor.b2 = zeros(1,aDim);
    AC.critic.W1 = randn(sDim, cfg.drl.hiddenDim)*0.01;
    AC.critic.b1 = zeros(1,cfg.drl.hiddenDim);
    AC.critic.W2 = randn(cfg.drl.hiddenDim,1)*0.01;
    AC.critic.b2 = 0;
    AC.cfg = cfg;
end
