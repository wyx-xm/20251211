% InitVortexEnvironment.m
% 生成 Rankine Vortex 种子（涡心+强度+半径）
function seed = InitVortexEnvironment(sz)
    % grid: 51x51 by default (for a 1000 map with 20m cell)
    [X, Y] = meshgrid(linspace(0, sz, 51));
    seed.X = X; seed.Y = Y;
    nv = 5;
    rng(1);
    seed.vortices = zeros(nv,4);
    seed.vortices(:,1:2) = rand(nv,2) * sz;      % cx, cy
    seed.vortices(:,3) = (rand(nv,1)-0.5)*2000;  % Gamma
    seed.vortices(:,4) = 50 + rand(nv,1)*100;    % R
end
