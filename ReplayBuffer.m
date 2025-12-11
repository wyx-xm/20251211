% ReplayBuffer.m
% 简单经验池（对象风格，支持 buffer = buffer.push(...) 链式调用）
function buf = ReplayBuffer(cfg)
    % 内部数据存储
    data.s = []; data.a = []; data.r = []; data.done = []; data.logp = []; data.v = [];

    % 方法句柄
    buf.push = @push;
    buf.sample = @sample;
    buf.clear = @clearBuf;
    buf.size = @getSize;

    % push: 添加样本，返回方法句柄结构（链式）
    function nb = push(s,a,r,d,logp,v)
        % 确保行向量/列向量一致
        if isrow(s), s = s'; end
        % Append (处理不同 shape 的 s)
        if isempty(data.s)
            data.s = s(:)'; % store as row
        else
            % ensure same width: pad or trim if necessary
            if size(s,2) ~= size(data.s,2)
                % try to flatten/reshape
                s = s(:)';
                if size(s,2) ~= size(data.s,2)
                    % pad with zeros
                    s = [s, zeros(1, size(data.s,2)-size(s,2))];
                end
            end
            data.s = [data.s; s];
        end
        data.a = [data.a; a];
        data.r = [data.r; r];
        data.done = [data.done; d];
        data.logp = [data.logp; logp];
        data.v = [data.v; v];
        nb = buf; % return struct with methods
    end

    % sample: 随机采样 n 条
    function batch = sample(n)
        N = size(data.s,1);
        n = min(n, N);
        idx = randperm(N, n);
        batch.s = data.s(idx,:);
        batch.a = data.a(idx,:);
        batch.r = data.r(idx,:);
        batch.logp = data.logp(idx,:);
        batch.v = data.v(idx,:);
    end

    function clearBuf()
        data.s = []; data.a = []; data.r = []; data.done = []; data.logp = []; data.v = [];
    end

    function n = getSize()
        n = size(data.s,1);
    end
end
