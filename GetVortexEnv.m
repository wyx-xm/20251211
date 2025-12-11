% GetVortexEnv.m
% 根据种子和时间生成流场 U,V 矩阵
function env = GetVortexEnv(seed, t, sz)
    X = seed.X; Y = seed.Y;
    U = zeros(size(X)); V = zeros(size(Y));
    for i=1:size(seed.vortices,1)
        cx = seed.vortices(i,1) + 50*sin(t/500 + i);
        cy = seed.vortices(i,2) + 50*cos(t/500 + i);
        Gamma = seed.vortices(i,3);
        R = seed.vortices(i,4);
        dx = X - cx; dy = Y - cy;
        r2 = dx.^2 + dy.^2; r = sqrt(r2);
        mask = r < R;
        vel = zeros(size(r));
        vel(mask) = Gamma .* r(mask) ./ (2*pi*R^2);
        vel(~mask) = Gamma ./ (2*pi*(r(~mask)+1e-6));
        U = U - vel .* dy ./ (r+1e-6);
        V = V + vel .* dx ./ (r+1e-6);
    end
    U = U + 0.5; % background flow
    env.X=X; env.Y=Y; env.U=U; env.V=V;
end
