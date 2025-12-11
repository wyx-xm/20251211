% ActorCritic_policy.m
% 给定 state s，Actor 输出 action (discrete choice) 与 logp, Critic 输出 v
function [action, logp, v] = ActorCritic_policy(AC, s)
    % actor forward
    z = tanh(s * AC.actor.W1 + AC.actor.b1);
    logits = z * AC.actor.W2 + AC.actor.b2;
    probs = softmax(logits,2);
    % sample discrete action
    action = categorical_sample(probs);
    logp = log(probs(action)+1e-12);
    % critic
    zc = tanh(s * AC.critic.W1 + AC.critic.b1);
    v = zc * AC.critic.W2 + AC.critic.b2;
end

function a = categorical_sample(probs)
    r = rand();
    cs = cumsum(probs);
    a = find(cs >= r,1,'first');
end
