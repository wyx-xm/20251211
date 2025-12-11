% EvolveWithAuctionAndFlow.m
% 拍卖 + 流场感知进化算子（与 GNN attention 结合）
function pop = EvolveWithAuctionAndFlow(pop, fitness, attMatrix, cfg, env, scene, gnnWeights)
    popSize = length(pop);
    % elitism
    [~, idxs] = sort(fitness);
    eliteCount = max(1, round(0.1*popSize));
    newPop = pop;
    for i=1:eliteCount
        newPop(i) = pop(idxs(i));
    end
    % rest via auction crossover + mutation
    for i=eliteCount+1:popSize
        % parent selection (tournament)
        p1 = pop(idxs(randi(round(popSize*0.3))));
        p2 = pop(idxs(randi(round(popSize*0.3))));
        child = p1;
        % auction over a subset of tasks
        auctionTasks = randperm(cfg.numTargets, max(3, round(cfg.numTargets*0.2)));
        for t = auctionTasks
            bids = zeros(1,cfg.numAUVs);
            for k=1:cfg.numAUVs
                % use simple flow-aware cost: dist*(1 - 0.3*flowDot)
                dist = norm(scene.targets(t,:) - cfg.depot);
                mid = round((scene.targets(t,:) + cfg.depot)/2);
                ix = max(1,min(size(env.U,2), round(mid(1)/20)+1));
                iy = max(1,min(size(env.U,1), round(mid(2)/20)+1));
                u = env.U(iy,ix); v = env.V(iy,ix);
                vec = scene.targets(t,:) - cfg.depot;
                flowDot = dot(vec/norm(vec+1e-6), [u,v]);
                bids(k) = dist * (1 - 0.5*flowDot) + randn*5;
            end
            [~, winner] = min(bids);
            child.assign(t) = winner;
        end
        % mutation: small random swaps
        if rand < 0.3
            idxsSwap = randperm(cfg.numTargets, 2);
            tmp = child.assign(idxsSwap(1));
            child.assign(idxsSwap(1)) = child.assign(idxsSwap(2));
            child.assign(idxsSwap(2)) = tmp;
        end
        newPop(i) = child;
    end
    pop = newPop;
end
