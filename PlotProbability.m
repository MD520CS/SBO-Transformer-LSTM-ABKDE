%% 闲鱼：深度学习与智能算法
%% 唯一官方店铺：https://mbd.pub/o/author-aWWbm3BtZw==
%% 微信公众号：强盛机器学习，关注公众号获得更多免费代码！
function PlotProbability(TestOutputs,RealOutputs,Num_Fcst,Lower,Upper,llimit,rlimit,IColor)
    %% 区间估计
    figure; %概率绘图
    x = 1:Num_Fcst;
    hold on
    n = [0.2;0.25;0.3;0.35;0.4;0.45;0.5]; %区间透明度
    for j = 1:7
        window(j)=fill([x,fliplr(x)],[Lower(:,j)',fliplr(Upper(:,j)')],IColor,'FaceAlpha',n(j));
        window(j).EdgeColor = 'none';
        hold on
        plot(Upper(:,j),'Marker',"none","LineStyle","none","Tag",'none',"Visible","off");
        hold on
        plot(Lower(:,j),'Marker',"none","LineStyle","none","Tag",'none',"Visible","off");
        hold on
    end
    plot(RealOutputs, 'r*','MarkerSize',4);
    hold on
    plot(TestOutputs,'r-','LineWidth',1.5);
    hold on
    xlim([llimit rlimit]);
    ylim([min(min(Lower)) max(max(Upper))]);
    xlabel('采样点',"FontSize",13);
    ylabel('预测结果',"FontSize",13);
    legend('95%置信区间',"","",'90%置信区间',"","",'75%置信区间',"","",...
    '50%置信区间',"","",'25%置信区间',"","",'10%置信区间',"","",...
    '5%置信区间',"","",'真实值','预测值','Location','best');
    grid on
    set(gcf,'color','w')
end

