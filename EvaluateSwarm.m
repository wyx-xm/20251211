% EvaluateSwarm.m
% 评估整个种群。对每个个体，使用 predictor 估计时间/能耗/安全值。
function [rawCosts, safetyVio, energyCons, paths] = EvaluateSwarm(pop, cfg, env, scene, predictor)
    N = length(pop);
    rawCosts = zeros(N,1);
    safetyVio = zeros(N,3); % three constraint types
    energyCons = zeros(N,1);
    paths = cell(N,1);
    for i=1:N
        totalCost = 0; totalVio = zeros(1,3); totalE = 0;
        pDetails = cell(cfg.numAUVs,1);
        for k=1:cfg.numAUVs
            taskIdx = pop(i).seq(pop(i).assign==k);
            if isempty(taskIdx)
                continue;
            end
            routePts = [cfg.depot; scene.targets(taskIdx,:); cfg.depot];
            % estimate cost using predictor per leg
            legCost = 0; legVio = zeros(1,3); legE = 0;
            traj = [];
            for s=1:size(routePts,1)-1
                p1 = routePts(s,:); p2 = routePts(s+1,:);
                dist = norm(p2-p1);
                % sample env along straight line for predictor
                samples = sampleFlowAlongLine(env, p1, p2, predictor.window);
                [tPred, sigma] = Predictor_predict(predictor, samples, dist, atan2(p2(2)-p1(2), p2(1)-p1(1)), mean(scene.priorities));
                % approximate energy cost: P ~ thrust * speed; use simple model
                speed = max(0.5, dist / (tPred+1e-6));
                power = 40 * speed; % crude
                eCost = power * tPred;
                legE = legE + eCost;
                % safety violation proxy: if path passes near danger center (500,500)
                mid = (p1+p2)/2;
                d2danger = norm(mid - [500,500]);
                vioCollision = max(0, 100 - d2danger); % larger if closer
                vioEnergy = max(0, eCost - cfg.phys.batteryCap*0.001); % scale
                vioTime = max(0, tPred - 300); % late if predicted > 300s
                legVio = legVio + [vioCollision, vioEnergy, vioTime];
                legCost = legCost + tPred; % cost measured as time
                % quick traj sample
                traj = [traj; linspace(p1(1),p2(1),5)', linspace(p1(2),p2(2),5)'];
            end
            totalCost = totalCost + legCost;
            totalVio = totalVio + legVio;
            totalE = totalE + legE;
            pDetails{k} = traj;
        end
        rawCosts(i) = totalCost;
        safetyVio(i,:) = totalVio;
        energyCons(i) = totalE;
        paths{i} = pDetails;
    end
end

function samples = sampleFlowAlongLine(env, p1, p2, k)
    % 在两点之间均匀采样流速场
    xs = linspace(p1(1), p2(1), k);
    ys = linspace(p1(2), p2(2), k);
    samples = zeros(k,2);
    for i=1:k
        ix = max(1, min(size(env.U,2), round(xs(i)/20)+1));
        iy = max(1, min(size(env.U,1), round(ys(i)/20)+1));
        samples(i,:) = [env.U(iy,ix), env.V(iy,ix)];
    end
end
