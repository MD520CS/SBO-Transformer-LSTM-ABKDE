%% SBO-Transformer-LSTM-ABKDE点预测与区间预测
% 修正内容：
% 1. 修复lstmLayer隐藏节点数必须为正整数的问题
% 2. 修复PlotKernelDensity真实输出行列方向错误
% 3. 修复采样点超过测试集样本数的问题
% 4. 修复预测结果反归一化维度问题
% 5. 修复CRPS输入维度问题
% 6. 增加数据和参数有效性检查

%% 清空环境
warning off
close all
clear
clc

rng(2026, 'twister');

%% 导入数据
dataFile = '数据集.xlsx';

if ~isfile(dataFile)
    error('未找到数据文件：%s，请将数据文件放在当前工作文件夹。', dataFile);
end

res = readmatrix(dataFile);

% 删除全部为空的行
res = res(~all(isnan(res), 2), :);

% 删除含有NaN或Inf的行
validRows = all(isfinite(res), 2);

if any(~validRows)
    fprintf('检测到%d行无效数据，已自动删除。\n', sum(~validRows));
    res = res(validRows, :);
end

if isempty(res)
    error('数据集为空或所有数据均无效。');
end

if size(res, 2) < 2
    error('数据集至少需要两列：输入变量和输出变量。');
end

%% 数据分析
num_size = 0.7;                     % 训练集比例
outdim = 1;                         % 输出变量数量
num_samples = size(res, 1);         % 样本总数

% 时序预测通常不打乱数据
% res = res(randperm(num_samples), :);

num_train_s = round(num_size * num_samples);

% 保证训练集和测试集均不为空
num_train_s = max(2, num_train_s);
num_train_s = min(num_samples - 1, num_train_s);

% 输入特征维度
f_ = size(res, 2) - outdim;

if f_ < 1
    error('没有检测到有效输入特征。');
end

fprintf('========================================\n');
fprintf('数据集信息\n');
fprintf('========================================\n');
fprintf('总样本数：%d\n', num_samples);
fprintf('训练样本数：%d\n', num_train_s);
fprintf('测试样本数：%d\n', num_samples - num_train_s);
fprintf('输入特征数：%d\n', f_);
fprintf('输出变量数：%d\n', outdim);
fprintf('========================================\n\n');

%% 划分训练集和测试集
P_train = res(1:num_train_s, 1:f_)';
T_train_original = res(1:num_train_s, f_ + 1:end)';

P_test = res(num_train_s + 1:end, 1:f_)';
T_test_original = res(num_train_s + 1:end, f_ + 1:end)';

M = size(P_train, 2);
N = size(P_test, 2);

if M < 2
    error('训练样本数量不足。');
end

if N < 1
    error('测试样本数量不足。');
end

%% 数据归一化
[p_train, ps_input] = mapminmax(P_train, 0, 1);
p_test = mapminmax('apply', P_test, ps_input);

[t_train_norm, ps_output] = mapminmax(T_train_original, 0, 1);
t_test_norm = mapminmax('apply', T_test_original, ps_output);

%% 转换为序列网络Cell格式
vp_train = cell(M, 1);
vp_test = cell(N, 1);

for i = 1:M
    vp_train{i, 1} = reshape(p_train(:, i), f_, 1);
end

for i = 1:N
    vp_test{i, 1} = reshape(p_test(:, i), f_, 1);
end

%% 输出数据格式转换
% trainNetwork的回归响应要求为：
% 样本数 × 输出变量数
t_train = t_train_norm';
t_test = t_test_norm';

% 原始尺度真实值，统一转换成列向量
T_train_real = T_train_original';
T_test_real = T_test_original';

T_train_real = double(T_train_real(:));
T_test_real = double(T_test_real(:));

%% 创建待优化函数
OptimizationEpochs = 60;

ObjFcn = @(x) CostFunctionFixed( ...
    x, ...
    vp_train, ...
    t_train, ...
    f_, ...
    outdim, ...
    OptimizationEpochs);

%% SBO优化参数设置
SearchAgents = 6;
Max_iterations = 10;

% 参数依次为：
% 1. 注意力头数
% 2. LSTM隐藏节点数
% 3. 初始学习率
% 4. L2正则化系数
lb = [2, 10, 1e-5, 1e-5];
ub = [10, 100, 1e-2, 1e-2];

dim = 4;

%% SBO优化
fprintf('开始执行SBO参数优化……\n');

[Best_score, Best_pos, Curve] = SBO( ...
    SearchAgents, ...
    Max_iterations, ...
    lb, ...
    ub, ...
    dim, ...
    ObjFcn);

fprintf('SBO参数优化完成。\n\n');

%% 检查SBO结果
if isempty(Best_pos)
    error('SBO未返回有效的最优参数。');
end

Best_pos = double(Best_pos(:)');

if numel(Best_pos) < dim
    error('SBO返回的最优参数数量不足。');
end

if any(~isfinite(Best_pos(1:dim)))
    error('SBO返回的最优参数包含NaN或Inf。');
end

%% 获得最优参数
Headcounts = round(Best_pos(1));
Headcounts = max(round(lb(1)), Headcounts);
Headcounts = min(round(ub(1)), Headcounts);

NumOfUnits = round(Best_pos(2));
NumOfUnits = max(round(lb(2)), NumOfUnits);
NumOfUnits = min(round(ub(2)), NumOfUnits);

InitialLearnRate = Best_pos(3);
InitialLearnRate = max(lb(3), InitialLearnRate);
InitialLearnRate = min(ub(3), InitialLearnRate);

L2Regularization = Best_pos(4);
L2Regularization = max(lb(4), L2Regularization);
L2Regularization = min(ub(4), L2Regularization);

fprintf('========================================\n');
fprintf('SBO最优参数\n');
fprintf('========================================\n');
fprintf('最优适应度：%.10f\n', Best_score);
fprintf('注意力头数：%d\n', Headcounts);
fprintf('LSTM隐藏节点数：%d\n', NumOfUnits);
fprintf('初始学习率：%.10f\n', InitialLearnRate);
fprintf('L2正则化系数：%.10f\n', L2Regularization);
fprintf('========================================\n\n');

%% 网络参数
numChannels = f_;
maxPosition = 256;
numHeads = Headcounts;
numKeyChannels = numHeads * 32;

fprintf('注意力键通道数：%d\n\n', numKeyChannels);

%% 创建网络
lgraph = CreateTransformerLSTMNetwork( ...
    numChannels, ...
    outdim, ...
    Headcounts, ...
    NumOfUnits, ...
    maxPosition);

%% 最终网络训练参数
FinalEpochs = 200;

options = trainingOptions('adam', ...
    'MaxEpochs', FinalEpochs, ...
    'MiniBatchSize', min(32, M), ...
    'GradientThreshold', 1, ...
    'InitialLearnRate', InitialLearnRate, ...
    'L2Regularization', L2Regularization, ...
    'Shuffle', 'every-epoch', ...
    'ExecutionEnvironment', 'cpu', ...
    'Verbose', false, ...
    'Plots', 'training-progress');

%% 训练最终模型
fprintf('开始训练最终Transformer-LSTM模型……\n');

rng(2026, 'twister');

net = trainNetwork( ...
    vp_train, ...
    t_train, ...
    lgraph, ...
    options);

fprintf('最终模型训练完成。\n\n');

%% 查看网络结构
analyzeNetwork(lgraph);

%% 模型预测
t_sim1_norm = predict( ...
    net, ...
    vp_train, ...
    'MiniBatchSize', min(32, M));

t_sim2_norm = predict( ...
    net, ...
    vp_test, ...
    'MiniBatchSize', min(32, N));

t_sim1_norm = double(t_sim1_norm);
t_sim2_norm = double(t_sim2_norm);

% 转换成样本数×输出维度
t_sim1_norm = reshape(t_sim1_norm, M, outdim);
t_sim2_norm = reshape(t_sim2_norm, N, outdim);

%% 数据反归一化
% mapminmax要求输出维度×样本数量
T_sim1 = mapminmax('reverse', t_sim1_norm', ps_output)';
T_sim2 = mapminmax('reverse', t_sim2_norm', ps_output)';

T_sim1 = double(T_sim1(:));
T_sim2 = double(T_sim2(:));

if numel(T_sim1) ~= M
    error('训练集预测值数量与训练样本数不一致。');
end

if numel(T_sim2) ~= N
    error('测试集预测值数量与测试样本数不一致。');
end

%% 绘制SBO进化曲线
figure;

plot(Curve, 'r-', 'LineWidth', 1.5);

xlabel('迭代次数');
ylabel('适应度');
legend('最佳适应度', 'Location', 'best');
title('SBO迭代曲线');
grid on;
set(gcf, 'Color', 'w');

%% 区间预测置信参数
z = [0.975; 0.95; 0.875; 0.75; 0.625; 0.55; 0.525];

numIntervals = length(z);

%% 计算训练集预测误差
% 这里计算训练集误差，因此样本数量必须使用M
Error = PlotError( ...
    T_sim1, ...
    T_train_real, ...
    M, ...
    '#3D59AB');

Error = double(Error(:)');

% 删除无效误差
Error = Error(isfinite(Error));

if isempty(Error)
    error('训练误差全部无效，无法进行ABKDE区间预测。');
end

%% 自适应带宽核密度估计
figure;

[y, t, optim_width, ~, ~, confb95] = ABKDE(Error);

y = double(y(:)');
t = double(t(:)');

% 检查置信区间维度
if size(confb95, 1) ~= 2 && size(confb95, 2) == 2
    confb95 = confb95';
end

if size(confb95, 1) ~= 2
    error('ABKDE返回的confb95尺寸错误，应为2×采样点数。');
end

if size(confb95, 2) ~= numel(t)
    error('ABKDE返回的置信区间长度与横坐标长度不一致。');
end

hold on;

window = fill( ...
    [t, fliplr(t)], ...
    [confb95(1, :), fliplr(confb95(2, :))], ...
    [7, 7, 7] / 8, ...
    'FaceAlpha', 0.5);

window.EdgeColor = 'none';

plot( ...
    t, ...
    confb95(1, :), ...
    'Color', [7, 7, 7] / 9, ...
    'LineWidth', 1);

plot( ...
    t, ...
    confb95(2, :), ...
    'Color', [7, 7, 7] / 9, ...
    'LineWidth', 1);

plot( ...
    t, ...
    y, ...
    'Color', [0.9, 0.2, 0.2], ...
    'LineWidth', 2);

[f0, xi0] = ksdensity(Error, 'Function', 'pdf');

plot( ...
    xi0, ...
    f0, ...
    'LineWidth', 1.5, ...
    'Color', "#7E2F8E");

xlim([min(t), max(t)]);

legend( ...
    {'95%置信核密度区域', ...
     '95%置信下界', ...
     '95%置信上界', ...
     '优化自适应带宽核密度估计曲线', ...
     '未优化固定带宽核密度估计曲线'}, ...
    'Location', 'best');

grid on;
ylabel('概率密度');
xlabel('预测误差');
title(sprintf('ABKDE误差概率密度估计，最优带宽为%.6f', optim_width));

set(gca, 'TickDir', 'out');
set(gcf, 'Color', 'w');

%% 计算误差分位数
Q1 = zeros(numIntervals, 1);
Q2 = zeros(numIntervals, 1);

for m = 1:numIntervals
    Q1(m) = QuantSol_FUN(t, y, 1 - z(m));
    Q2(m) = QuantSol_FUN(t, y, z(m));
end

%% 构造测试集预测区间
Lower = zeros(N, numIntervals);
Upper = zeros(N, numIntervals);

for m = 1:numIntervals
    Lower(:, m) = T_sim2 + Q1(m);
    Upper(:, m) = T_sim2 + Q2(m);
end

% 检查区间矩阵
if size(Lower, 1) ~= N || size(Upper, 1) ~= N
    error('预测区间行数与测试样本数量不一致。');
end

if size(Lower, 2) < 7 || size(Upper, 2) < 7
    error(['PlotKernelDensity函数需要至少7组预测区间，', ...
        '但当前Lower或Upper的列数不足7列。']);
end

%% 绘制概率预测结果
PlotProbability( ...
    T_sim2, ...
    T_test_real, ...
    numel(T_test_real), ...
    Lower, ...
    Upper, ...
    0, ...
    N, ...
    [0, 100, 0] / 255);

%% 绘制指定采样点核密度
% 原代码固定使用：
% time_index = [25; 50; 75; 100];
%
% 当测试集不足100个样本时会发生索引越界。
% 这里根据测试样本数量自动生成最多4个采样点。

if N >= 4
    time_index = round(linspace(1, N, 4))';
else
    time_index = (1:N)';
end

% 删除重复采样点
time_index = unique(time_index, 'stable');

% 限制在有效范围内
time_index = time_index(time_index >= 1 & time_index <= N);

% 再次转换成整数
time_index = round(time_index(:));

num_KD = numel(time_index);

if num_KD < 1
    error('没有可用于核密度绘图的有效采样点。');
end

%% 关键修正
% PlotKernelDensity内部使用：
% RealOutputs(TimeIndex(k),1)
%
% 因此RealOutputs必须为N×1列向量。
% 不能使用T_test_real'，否则会变成1×N行向量。
RealOutputsForKD = double(T_test_real(:));

% 尺寸检查
if size(RealOutputsForKD, 1) ~= N
    error('RealOutputsForKD的行数与测试样本数不一致。');
end

if max(time_index) > size(RealOutputsForKD, 1)
    error('核密度采样点超过真实输出数据的样本数量。');
end

if max(time_index) > size(Lower, 1)
    error('核密度采样点超过Lower预测区间的样本数量。');
end

if max(time_index) > size(Upper, 1)
    error('核密度采样点超过Upper预测区间的样本数量。');
end

fprintf('\n核密度绘图采样点为：');
fprintf('%d ', time_index);
fprintf('\n');

% 此处不能写T_test_real'
PlotKernelDensity( ...
    Lower, ...
    Upper, ...
    time_index, ...
    RealOutputsForKD, ...
    num_KD);

set(gcf, 'Color', 'w');

%% 区间预测指标计算
eta = 0.5;

[PINAW, mean_PINAW] = PINAW_FUN( ...
    Lower, ...
    Upper, ...
    T_test_real);

[PICP, mean_PICP] = PICP_FUN( ...
    Lower, ...
    Upper, ...
    T_test_real);

% CRPS输入：
% Lower为N×7
% T_sim2为N×1
% Upper为N×7
CRPS_input = [Lower, T_sim2, Upper];

[CRPS, mean_CRPS] = CRPS_FUN( ...
    CRPS_input, ...
    T_test_real);

[CWC, mean_CWC] = CWC_FUN( ...
    PINAW, ...
    PICP, ...
    eta, ...
    z);

fprintf('\n');
fprintf('========================================\n');
fprintf('区间预测指标\n');
fprintf('========================================\n');

for i = 1:numIntervals

    confidence_percent = (2 * z(i) - 1) * 100;

    fprintf('对于%.2f%%置信区间：\n', confidence_percent);
    fprintf('PINAW：%.6f\n', PINAW(i));
    fprintf('PICP ：%.6f\n', PICP(i));
    fprintf('CRPS ：%.6f\n', CRPS(i));
    fprintf('CWC  ：%.6f\n\n', CWC(i));

end

fprintf('平均PINAW：%.6f\n', mean_PINAW);
fprintf('平均PICP ：%.6f\n', mean_PICP);
fprintf('平均CRPS ：%.6f\n', mean_CRPS);
fprintf('平均CWC  ：%.6f\n', mean_CWC);
fprintf('========================================\n');

%% 点预测指标计算
predictionError = T_test_real - T_sim2;

SSE = sum(predictionError .^ 2);
SST = sum((T_test_real - mean(T_test_real)) .^ 2);

if SST > eps
    R2 = 1 - SSE / SST;
else
    R2 = NaN;
end

MAE2 = mean(abs(predictionError));

RMSE2 = sqrt(mean(predictionError .^ 2));

% 排除真实值为0的样本
validMAPE = abs(T_test_real) > eps;

if any(validMAPE)
    MAPE2 = mean(abs( ...
        predictionError(validMAPE) ./ ...
        T_test_real(validMAPE)));
else
    MAPE2 = NaN;
end

fprintf('\n');
fprintf('========================================\n');
fprintf('测试集点预测指标\n');
fprintf('========================================\n');
fprintf('R2：%.6f\n', R2);
fprintf('MAE：%.6f\n', MAE2);
fprintf('RMSE：%.6f\n', RMSE2);
fprintf('MAPE：%.6f\n', MAPE2);
fprintf('MAPE百分数：%.4f%%\n', MAPE2 * 100);
fprintf('========================================\n');

%% 绘制测试集真实值与预测值
figure;

plot(T_test_real, 'b-', 'LineWidth', 1.2);
hold on;
plot(T_sim2, 'r--', 'LineWidth', 1.2);

legend('真实值', '预测值', 'Location', 'best');
xlabel('测试样本');
ylabel('输出值');
title('测试集真实值与预测值对比');
grid on;
set(gcf, 'Color', 'w');

%% 绘制预测误差
figure;

plot(predictionError, 'k-', 'LineWidth', 1.2);
hold on;
yline(0, 'r--', 'LineWidth', 1);

xlabel('测试样本');
ylabel('预测误差');
title('测试集预测误差');
grid on;
set(gcf, 'Color', 'w');

%% 保存预测结果
ResultTable = table( ...
    (1:N)', ...
    T_test_real, ...
    T_sim2, ...
    predictionError, ...
    'VariableNames', ...
    {'SampleIndex', ...
     'TrueValue', ...
     'PredictedValue', ...
     'PredictionError'});

writetable( ...
    ResultTable, ...
    'SBO_Transformer_LSTM预测结果.xlsx');

%% 保存模型及变量
save( ...
    'SBO_Transformer_LSTM模型结果.mat', ...
    'net', ...
    'Best_score', ...
    'Best_pos', ...
    'Curve', ...
    'Headcounts', ...
    'NumOfUnits', ...
    'InitialLearnRate', ...
    'L2Regularization', ...
    'T_sim1', ...
    'T_sim2', ...
    'T_train_real', ...
    'T_test_real', ...
    'Lower', ...
    'Upper', ...
    'time_index', ...
    'R2', ...
    'MAE2', ...
    'RMSE2', ...
    'MAPE2', ...
    'PINAW', ...
    'PICP', ...
    'CRPS', ...
    'CWC', ...
    'ps_input', ...
    'ps_output');

fprintf('\n程序运行完成。\n');
fprintf('预测结果已保存至：SBO_Transformer_LSTM预测结果.xlsx\n');
fprintf('模型结果已保存至：SBO_Transformer_LSTM模型结果.mat\n');

%% ========================================================================
%% 局部函数1：SBO适应度函数
%% ========================================================================
function fitness = CostFunctionFixed( ...
    x, ...
    vp_train, ...
    t_train, ...
    numChannels, ...
    outdim, ...
    optimizationEpochs)

    fitness = inf;

    try
        %% 参数检查
        x = double(x(:)');

        if numel(x) < 4
            return;
        end

        if any(~isfinite(x(1:4)))
            return;
        end

        %% 注意力头数
        numHeads = round(x(1));
        numHeads = max(2, numHeads);
        numHeads = min(10, numHeads);

        %% LSTM隐藏节点数
        % lstmLayer的第一个参数必须为正整数
        numUnits = round(x(2));
        numUnits = max(10, numUnits);
        numUnits = min(100, numUnits);

        %% 初始学习率
        initialLearnRate = double(x(3));
        initialLearnRate = max(1e-5, initialLearnRate);
        initialLearnRate = min(1e-2, initialLearnRate);

        %% L2正则化系数
        l2Regularization = double(x(4));
        l2Regularization = max(1e-5, l2Regularization);
        l2Regularization = min(1e-2, l2Regularization);

        %% 划分内部训练集和验证集
        numSamples = numel(vp_train);

        if numSamples < 2
            return;
        end

        if numSamples >= 5

            numFit = floor(0.8 * numSamples);
            numFit = max(2, numFit);
            numFit = min(numSamples - 1, numFit);

            fitIndex = 1:numFit;
            valIndex = numFit + 1:numSamples;

        else

            fitIndex = 1:numSamples;
            valIndex = 1:numSamples;

        end

        XFit = vp_train(fitIndex);
        YFit = t_train(fitIndex, :);

        XValidation = vp_train(valIndex);
        YValidation = t_train(valIndex, :);

        %% 创建候选网络
        maxPosition = 256;

        candidateGraph = CreateTransformerLSTMNetwork( ...
            numChannels, ...
            outdim, ...
            numHeads, ...
            numUnits, ...
            maxPosition);

        %% 候选网络训练参数
        candidateOptions = trainingOptions('adam', ...
            'MaxEpochs', optimizationEpochs, ...
            'MiniBatchSize', min(32, numel(fitIndex)), ...
            'GradientThreshold', 1, ...
            'InitialLearnRate', initialLearnRate, ...
            'L2Regularization', l2Regularization, ...
            'Shuffle', 'every-epoch', ...
            'ExecutionEnvironment', 'cpu', ...
            'Verbose', false, ...
            'Plots', 'none');

        rng(2026, 'twister');

        %% 训练候选网络
        candidateNet = trainNetwork( ...
            XFit, ...
            YFit, ...
            candidateGraph, ...
            candidateOptions);

        %% 验证集预测
        validationPrediction = predict( ...
            candidateNet, ...
            XValidation, ...
            'MiniBatchSize', min(32, numel(valIndex)));

        validationPrediction = double(validationPrediction(:));
        validationTarget = double(YValidation(:));

        if numel(validationPrediction) ~= numel(validationTarget)
            fitness = inf;
            return;
        end

        if any(~isfinite(validationPrediction))
            fitness = inf;
            return;
        end

        %% 均方误差适应度
        errorValue = validationPrediction - validationTarget;

        fitness = mean(errorValue .^ 2);

        if isempty(fitness) || ~isfinite(fitness) || fitness < 0
            fitness = inf;
        end

    catch ME

        fprintf('当前候选参数训练失败：%s\n', ME.message);
        fitness = inf;

    end
end

%% ========================================================================
%% 局部函数2：创建Transformer-LSTM网络
%% ========================================================================
function lgraph = CreateTransformerLSTMNetwork( ...
    numChannels, ...
    outdim, ...
    numHeads, ...
    numUnits, ...
    maxPosition)

    %% 参数安全处理
    numChannels = round(double(numChannels));
    numChannels = max(1, numChannels);

    outdim = round(double(outdim));
    outdim = max(1, outdim);

    numHeads = round(double(numHeads));
    numHeads = max(1, numHeads);

    numUnits = round(double(numUnits));
    numUnits = max(1, numUnits);

    maxPosition = round(double(maxPosition));
    maxPosition = max(1, maxPosition);

    % 保证numKeyChannels能够被numHeads整除
    numKeyChannels = numHeads * 32;

    %% 网络层
    layers = [
        sequenceInputLayer( ...
            numChannels, ...
            'Normalization', 'none', ...
            'Name', 'input')

        positionEmbeddingLayer( ...
            numChannels, ...
            maxPosition, ...
            'Name', 'pos-emb')

        additionLayer( ...
            2, ...
            'Name', 'add')

        selfAttentionLayer( ...
            numHeads, ...
            numKeyChannels, ...
            'AttentionMask', 'causal', ...
            'Name', 'attention-causal')

        selfAttentionLayer( ...
            numHeads, ...
            numKeyChannels, ...
            'Name', 'attention-global')

        lstmLayer( ...
            numUnits, ...
            'OutputMode', 'sequence', ...
            'Name', 'lstm')

        reluLayer( ...
            'Name', 'relu')

        dropoutLayer( ...
            0.1, ...
            'Name', 'dropout')

        indexing1dLayer( ...
            'last', ...
            'Name', 'last-index')

        fullyConnectedLayer( ...
            outdim, ...
            'Name', 'fully-connected')

        regressionLayer( ...
            'Name', 'regression')
        ];

    %% 创建层图
    lgraph = layerGraph(layers);

    % 原始输入同时连接到加法层第二输入端口
    lgraph = connectLayers( ...
        lgraph, ...
        'input', ...
        'add/in2');
end