% Predictor_predict.m
% 基于占位 predictor 进行时间估计（快速、能运行）
% 输入：envSamples [k x 2], dist, heading, priority
function [t_pred, sigma] = Predictor_predict(predictor, envSamples, dist, heading, priority)
    % 简单推理：平均流场速度影响时间
    avgFlow = mean(sqrt(envSamples(:,1).^2 + envSamples(:,2).^2) + 1e-6);
    % baseline speed 2.5 m/s adjusted by flow (正顺减少时间)
    baseSpeed = 2.5;
    effSpeed = max(0.5, baseSpeed + 0.5*avgFlow); % 0.5 guard
    t_pred = dist / effSpeed + 0.1*priority; % add small priority factor
    sigma = max(1e-3, 0.05 * t_pred);
end
