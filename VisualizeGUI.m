% VisualizeGUI.m
% 简单 GUI，可在运行中刷新流场、任务点与最优轨迹
function gui = VisualizeGUI(cfg, scene)
    fig = figure('Name','DeepHRL AUV NextGen GUI','NumberTitle','off','Color','w','Position',[100 100 1200 700]);
    ax = axes(fig,'Position',[0.05 0.2 0.6 0.75]); hold(ax,'on'); axis(ax,'equal');
    quiv = []; pts = scatter(ax, scene.targets(:,1), scene.targets(:,2), scene.priorities*30, 'k','filled');
    title(ax,'Flow Field & AUV Trajectories');
    colormap(ax,'jet');
    cax = axes(fig,'Position',[0.7 0.55 0.25 0.35]); plot(cax, nan); title(cax,'Cost History'); grid(cax,'on');
    sfig = struct();
    sfig.fig = fig; sfig.ax = ax; sfig.quiv = quiv; sfig.pts = pts; sfig.costAx = cax;
    sfig.update = @(env, scene, bestSol, hist) updateGUI(sfig, env, scene, bestSol, hist, cfg);
    sfig.finalize = @() close(fig);
    gui = sfig;
end

function updateGUI(sfig, env, scene, bestSol, hist, cfg)
    ax = sfig.ax; cla(ax); hold(ax,'on');
    % draw flow
    step = 2;
    streamslice(ax, env.X, env.Y, env.U, env.V, step);
    % draw danger region
    viscircles(ax, [500,500], 100, 'Color','r','LineStyle','--');
    % tasks
    scatter(ax, scene.targets(:,1), scene.targets(:,2), scene.priorities*30, 'k','filled');
    % draw best paths if provided
    if ~isempty(bestSol) && isfield(bestSol,'seq')
        colors = lines(cfg.numAUVs);
        for k=1:cfg.numAUVs
            taskIdx = bestSol.seq(bestSol.assign==k);
            if isempty(taskIdx), continue; end
            pts = [cfg.depot; scene.targets(taskIdx,:); cfg.depot];
            plot(ax, pts(:,1), pts(:,2), '-o','Color',colors(k,:),'LineWidth',2);
        end
    end
    title(ax,'Flow Field & AUV Trajectories');
    % update cost plot
    figure(sfig.fig); axes(sfig.costAx); cla; plot(hist.cost,'LineWidth',2); title('Cost History'); grid on;
    drawnow;
end
