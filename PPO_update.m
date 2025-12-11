% PPO_update.m
% 简化的 PPO 更新实现（批量）
% batch must contain fields: s, a, r, logp_old, v_old
function AC = PPO_update(AC, batch, cfg)
    % unpack
    s = batch.s; a = batch.a; r = batch.r; logp_old = batch.logp; v_old = batch.v;
    % compute advantages (simple)
    returns = r; % for simplicity (no GAE)
    adv = returns - v_old;
    for ep=1:cfg.ppo.epochs
        % forward
        z = tanh(s * AC.actor.W1 + AC.actor.b1);
        logits = z * AC.actor.W2 + AC.actor.b2;
        probs = softmax_batch(logits);
        % gather logp for taken actions
        logp = arrayfun(@(i) log(probs(i,a(i))+1e-12), 1:size(s,1))';
        ratio = exp(logp - logp_old);
        clip = min(max(ratio, 1-cfg.ppo.clip), 1+cfg.ppo.clip);
        % policy loss
        L_clip = -mean(min(ratio .* adv, clip .* adv));
        % value loss
        zc = tanh(s * AC.critic.W1 + AC.critic.b1);
        vpred = zc * AC.critic.W2 + AC.critic.b2;
        L_v = mean((returns - vpred).^2);
        % simple gradient steps (SGD)
        % compute gradients numerical approx (very simplified)
        alpha = cfg.drl.learningRate;
        % actor params update (policy gradient ascent)
        % approximate: shift logits towards actions with positive advantage
        for i=1:length(a)
            AC.actor.W2(:,a(i)) = AC.actor.W2(:,a(i)) - alpha * L_clip * z(i,:)';
        end
        % critic update
        AC.critic.W2 = AC.critic.W2 - alpha * cfg.ppo.vfCoef * ( (mean(vpred)-mean(returns)) );
    end
    % Note: 该实现为非常简化的示意版，若需严格 PPO，请替换为 autograd-based 实现或使用 RL toolbox
end

function p = softmax_batch(logits)
    ex = exp(logits - max(logits,[],2));
    p = ex ./ sum(ex,2);
end
