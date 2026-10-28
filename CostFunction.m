function valError = CostFunction(optVars)

%% 赋值
Headcounts = abs(round(optVars(1)));   % 自注意力机制头数
NumOfUnits = abs(round(optVars(2)));   % 自注意力机制头数
InitialLearnRate = abs(optVars(3));    % 初始学习率
L2Regularization  = abs(optVars(4));   % 正则化参数

%%  获取数据
vp_train = evalin('base', 'vp_train');
t_train = evalin('base', 't_train');
f_ = evalin('base', 'f_');
outdim = evalin('base', 'outdim');

%%  网络参数设置
numChannels = f_;                % 输入通道数
maxPosition = 256;               % 最大位置编码长度
numHeads = Headcounts;           % 自注意力机制中的头数
numKeyChannels = numHeads * 32;  % 每个头的键的通道数，总共 numHeads 个头，每个头 32 个通道

%%  创建网络
layers = [ 
    sequenceInputLayer(numChannels, Name="input")                           % 序列输入层
    positionEmbeddingLayer(numChannels, maxPosition, Name="pos-emb")        % 位置嵌入层
    additionLayer(2, Name="add")                                            % 加法层
    selfAttentionLayer(numHeads, numKeyChannels, 'AttentionMask', 'causal') % 自注意力层，使用因果掩码
    selfAttentionLayer(numHeads, numKeyChannels)                            % 自注意力层
    lstmLayer(NumOfUnits)                                                   % LSTM层
    reluLayer                                                               % Relu激活层
    dropoutLayer(0.1)                                                       % 丢弃层
    indexing1dLayer("last")                                                 % 索引层，选择最后一个时间步的数据
    fullyConnectedLayer(outdim)                                             % 全连接层
    regressionLayer];                                                       % 回归层

lgraph = layerGraph(layers);                                                % 创建层图对象
lgraph = connectLayers(lgraph, "input", "add/in2");                         % 将输入层和加法层的第二个输入端口连接

%%  训练参数设置
options = trainingOptions('adam', ...                 % 优化算法Adam
    'MaxEpochs', 100, ...                             % 最大训练次数
    'GradientThreshold', 1, ...                       % 梯度阈值
    'InitialLearnRate', InitialLearnRate, ...         % 初始学习率
    'L2Regularization', L2Regularization, ...         % 正则化系数
    'Shuffle', 'every-epoch', ...                     % 每次训练打乱数据集
    'ExecutionEnvironment', 'cpu',...                 % 训练环境
    'Verbose', 0, ...                                 % 开启/关闭优化过程
    'Plots', 'none');                                 % 是否画出曲线

%%  训练网络
net = trainNetwork(vp_train, t_train, lgraph, options);

%%  得到网络预测值
t_sim1 = predict(net, vp_train); 

%%  计算误差
valError = sqrt(double(mse(t_train, t_sim1)));

end