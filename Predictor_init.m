% Predictor_init.m
% 初始化 CNN-LSTM 预测器结构（占位实现）
% 重要说明：此处实现为可以直接运行的占位版。若需高精度预测，需用真实仿真数据训练 CNN-LSTM。
function predictor = Predictor_init(cfg)
    predictor.window = cfg.pred.window;
    predictor.cnnFilters = cfg.pred.cnnFilters;
    predictor.lstmHidden = cfg.pred.lstmHidden;
    % 随机初始化一些权重（用于演示推理）
    rng(3);
    predictor.W_conv = randn(predictor.window, predictor.cnnFilters(1))*0.01;
    predictor.W_lstm = randn(predictor.lstmHidden, predictor.cnnFilters(end))*0.01;
    predictor.W_fc = randn(predictor.lstmHidden + 3, 1)*0.01; % + dist, heading, priority
    predictor.norm.mean = 0; predictor.norm.std = 1;
end
