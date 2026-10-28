%% 闲鱼：深度学习与智能算法
%% 唯一官方店铺：https://mbd.pub/o/author-aWWbm3BtZw==
%% 微信公众号：强盛机器学习，关注公众号获得更多免费代码！
function [PINAW,mean_PINAW] = PINAW_FUN(Lower,Upper,Real)
    PINAW = sum(Upper-Lower)/(length(Real)*(max(Real)-min(Real)));
    mean_PINAW = mean(PINAW);
end

