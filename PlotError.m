function [Error]=PlotError(TestOutputs,RealOutputs,N,TColor)

    figure;
    Error=RealOutputs-TestOutputs;
    Error_Mean=mean(Error);
    Error_Std=std(Error);
    MSE=mean(Error.^2); %均方误差
    RMSE=sqrt(MSE); %均方根误差
    
    subplot(2,2,[1 2]);
    plot(RealOutputs,'Color','#D95319','LineWidth',1);
    hold on;
    plot(TestOutputs,'--','Color',TColor,'LineWidth',1);
    legend('真实值','预测值');
    xlabel('采样点');
    ylabel('预测结果');
    xlim([0,N]);
    grid on;

    
    subplot(2,2,3);
    plot(Error,':k','LineWidth',2);
    xlabel('采样点');
    legend('残差');
    title(['MSE = ' strcat(num2str(MSE)) ', RMSE = ' strcat(num2str(RMSE))]);
    xlim([0,N]);
    grid on;

    
    subplot(2,2,4);
    h=histfit(Error,60,'kernel');
    h(1).FaceColor = [0.8 0.8 1];
    h(2).Color = 'b';
    xlabel('残差');
    ylabel('个数');
    title(['残差平均值 = ' strcat(num2str(Error_Mean)) ', 残差标准差 = ' strcat(num2str(Error_Std))]);
    set(gcf,'color','w')
end

