% SimulateAUV6DOF.m
% 近似 6-DOF 仿真：用于对少数候选进行精算（节省 CPU）
% 为了效率，这里采用简化动力学（surge + yaw 动力学）
function [T, E, Vio, Traj] = SimulateAUV6DOF(routePts, cfg, env)
    state = zeros(12,1); % [x,y,z,phi,theta,psi, u,v,w, p,q,r]
    state(1:2) = routePts(1,:);
    dt = 1.0; T=0; E=0; Vio=0; Traj=[];
    for seg=1:size(routePts,1)-1
        target = routePts(seg+1,:);
        dist = norm(target - state(1:2)');
        while dist > 5
            ix = max(1,min(size(env.U,2), round(state(1)/20)+1));
            iy = max(1,min(size(env.U,1), round(state(2)/20)+1));
            u_c = env.U(iy,ix); v_c = env.V(iy,ix);
            los_psi = atan2(target(2)-state(2), target(1)-state(1));
            psi_err = angdiff(state(6), los_psi);
            desired_r = 1.0 * psi_err;
            desired_u = 2.5;
            thrust = min(cfg.phys.maxThrust, 40 + 10*randn*0.01);
            drag = 10 * state(7) * abs(state(7));
            state(7) = max(0, state(7) + (thrust - drag)/cfg.phys.mass * dt);
            state(12) = desired_r;
            state(6) = state(6) + state(12) * dt;
            x_dot = state(7) * cos(state(6)) + u_c;
            y_dot = state(7) * sin(state(6)) + v_c;
            state(1) = state(1) + x_dot * dt;
            state(2) = state(2) + y_dot * dt;
            power = thrust * state(7) + 10;
            E = E + power * dt;
            distToDanger = norm(state(1:2) - [500,500]);
            if distToDanger < 100
                Vio = Vio + (100 - distToDanger);
            end
            T = T + dt;
            Traj = [Traj; state(1:2)'];
            dist = norm(target - state(1:2)');
            if T > 5000, break; end
        end
    end
end
