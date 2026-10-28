%% 闲鱼：深度学习与智能算法
%% 唯一官方店铺：https://mbd.pub/o/author-aWWbm3BtZw==
%% 微信公众号：强盛机器学习，关注公众号获得更多免费代码！

function PlotKernelDensity(Lower,Upper,TimeIndex,RealOutputs,Num)
%% 核密度估计
    
    %采样数据矩阵
    for k = 1:Num
        sampoint(:,k) = [Lower(TimeIndex(k),1);Lower(TimeIndex(k),2);Lower(TimeIndex(k),3);Lower(TimeIndex(k),4);Lower(TimeIndex(k),5);...
        Lower(TimeIndex(k),6);Lower(TimeIndex(k),7);RealOutputs(TimeIndex(k),1);Upper(TimeIndex(k),1);Upper(TimeIndex(k),2);Upper(TimeIndex(k),3);...
        Upper(TimeIndex(k),4);Upper(TimeIndex(k),5);Upper(TimeIndex(k),6);Upper(TimeIndex(k),7)];
    end
    for l = 1:Num
        [f(l,:),xi(l,:)] = ksdensity(sampoint(:,l),'Function','pdf'); %核密度估计
    end
    figure;
    hold on
    vbls = string(zeros(1,size(TimeIndex,1)));%采样点标签
    for io = 1:size(TimeIndex,1)
        vbls(1,io) = sprintf('采样点%d',TimeIndex(io));
    end
    for v = 1:Num
        ax(v) = subplot(sqrt(Num),sqrt(Num),v);
        h = gca;
        hold on
        window(:,v)=fill([xi(v,:)',fliplr(xi(v,:)')],[f(v,:)',fliplr(f(v,:)')],[0 100 0]/255,'FaceAlpha',0.2);
        window(v).EdgeColor = 'none';
        hold on
        plot(xi(v,:),f(v,:),'LineWidth',1.2,'Color','#0000FF');
        hold on
        stem(RealOutputs(TimeIndex(v,1),1),max(f(v,:)),'Marker',"none",'LineWidth',1.2,'Color','#D95319');
        hold on
        legend({'','',strcat('核密度曲线'),'实际值'},'Location','northoutside');
        ax(v).YLabel.String = '概率密度';
        ax(v).YLabel.FontName = '仿宋';
        ax(v).XLabel.String = '预测结果';
        ax(v).Title.String = vbls(v);
        ax(v).FontSize = 11;
        ax(v).YLim = [min(f(v,:)) max(f(v,:))];
        ax(v).XLim = [xi(v,1) xi(v,end)];
    end
end

