%% 闲鱼：深度学习与智能算法
%% 唯一官方店铺：https://mbd.pub/o/author-aWWbm3BtZw==
%% 微信公众号：强盛机器学习，关注公众号获得更多免费代码！
function [CWC,mean_CWC] = CWC_FUN(PINAW,PICP,Eta,Beta)
   % CWC
    PINAW = PINAW';
    PICP = PICP';
    for i = 1:numel(PINAW)
        if PICP(i) < Beta(i)
            Gamma(i) = 1;
        else 
            Gamma(i) = 0;
        end
    end
    for m = 1:numel(PINAW)
        CWC(m) = PINAW(m)+Gamma(m)*exp(-Eta*(PICP(m)-Beta(m)));
    end
    mean_CWC = mean(CWC);
end

