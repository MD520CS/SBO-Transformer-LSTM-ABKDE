%% 闲鱼：深度学习与智能算法
%% 唯一官方店铺：https://mbd.pub/o/author-aWWbm3BtZw==
%% 微信公众号：强盛机器学习，关注公众号获得更多免费代码！
function Q = QuantSol_FUN(X,Y,m)
    S = cumtrapz(X,Y);
    Index = find(abs(m-S)<=1e-2);
    Q = X(max(Index));
end

