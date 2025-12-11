% GATForward.m
% 简化版 GAT forward：将 job nodes 和 robot nodes 投影到 embedding 空间
% 并计算 job->robot 的 attention scores（含边特征）
function [jobEmb, robEmb, edgeScores] = GATForward(jobFeat, robFeat, edgeFeat, w)
    % jobFeat: [M x fj], robFeat: [N x fr], edgeFeat: [M*N x fe]
    M = size(jobFeat,1);
    N = size(robFeat,1);
    d = size(w.W_job,2);

    Jproj = jobFeat * w.W_job + repmat(w.b_job, M, 1);
    Rproj = robFeat * w.W_robot + repmat(w.b_robot, N, 1);
    Eproj = edgeFeat * w.W_edge + repmat(w.b_edge, size(edgeFeat,1), 1);

    jobEmb = tanh(Jproj); robEmb = tanh(Rproj);

    edgeScores = zeros(M,N);
    idx = 1;
    for j=1:M
        for r=1:N
            concat = [jobEmb(j,:), robEmb(r,:), Eproj(idx,:)];
            edgeScores(j,r) = leakyrelu(concat * w.W_att);
            idx = idx + 1;
        end
    end
    % optionally normalize per-job
    edgeScores = softmax(edgeScores,2);
end

function y = leakyrelu(x)
    y = max(0.01*x, x);
end
