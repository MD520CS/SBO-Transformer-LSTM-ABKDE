%% 闲鱼：深度学习与智能算法
%% 唯一官方店铺：https://mbd.pub/o/author-aWWbm3BtZw==
%% 微信公众号：强盛机器学习，关注公众号获得更多免费代码！
function [CRPS,mean_CRPS] = CRPS_FUN(probal_data,obs_data)
    % CRPS
    for i =1:size(probal_data,2)
           [Empf(:,i),xval(:,i)] = ecdf(probal_data(:,i));
    end
    %%
    Empf(end,:) = [];
    xval(end,:) = [];
    %%
    for i =1:size(probal_data,2)
        for j = 1:size(xval,1)
            if xval(j,i) < obs_data(j)
                Heaviside(j,i) = 0;
            else 
                Heaviside(j,i) = 1;
            end
        end    
    end
    CRPS = sum((Empf-Heaviside).^2)/length(obs_data);
    mean_CRPS = mean(CRPS);
end

