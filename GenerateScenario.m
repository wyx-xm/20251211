function scene = GenerateScenario(cfg)
% GenerateScenario.m
% 修复版：必须生成 targets 和 auvInitPos（与 cfg.numAUVs 匹配）

    % --- 1) 生成任务点 ---
    scene.targets = cfg.mapSize * rand(cfg.numTargets, 2);
    scene.priorities = randi([1, 5], cfg.numTargets, 1);

    % --- 2) 生成 AUV 初始位置（必须匹配 cfg.numAUVs）---
    % 环绕 depot 均匀分布（减少重叠）
    N = cfg.numAUVs;
    ang = linspace(0, 2*pi, N+1); 
    ang(end) = [];
    radius = 50;  % 距离 depot 的半径

    auvInitPos = zeros(N,2);
    for i = 1:N
        auvInitPos(i,:) = cfg.depot + radius*[cos(ang(i)), sin(ang(i))];
    end

    scene.auvInitPos = auvInitPos;

end
