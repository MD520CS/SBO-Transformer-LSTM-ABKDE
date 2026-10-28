%% 闲鱼：深度学习与智能算法
%% 唯一官方店铺：https://mbd.pub/o/author-aWWbm3BtZw==
%% 微信公众号：强盛机器学习，关注公众号获得更多免费代码！
function [y,t,optw,gs,C,confb95,yb] = ABKDE(x,tin)

%参数初始化

M = 80;            %带宽优化次数

WinFunc = 'Gauss'; %核函数 ('Gauss','Laplace','Cauchy') 默认使用鲁棒性最好的高斯核

bootsamp_num = 1*1e2;        %每批样本迭代次数（采用bootstrap抽样法）

x = reshape(x,1,numel(x));

if nargin == 1 %没有给定待估计点
    T = max(x) - min(x);
    [mbuf,nbuf,dt_samp] = find( sort(diff(sort(x))),1,'first');
    tin = linspace(min(x),max(x), min(ceil(T/dt_samp),1e3)); %生成待估计样本点
    t = tin;
    x_ab = x( logical((x >= min(tin)) .*(x <= max(tin))) ) ;
else %给定待估计点
    T = max(tin) - min(tin);
    x_ab = x( logical((x >= min(tin)) .*(x <= max(tin))) ) ;
    [mbuf,nbuf,dt_samp] = find( sort(diff(sort(x_ab))),1,'first');

    if dt_samp > min(diff(tin))
        t = linspace(min(tin),max(tin), min(ceil(T/dt_samp),1e3));
    else
        t = tin;
    end
end
clear mbuf nbuf;
dt = min(diff(t));


y_hist = histc(x_ab,t-dt/2)/dt; %#ok<*HISTC>
L = length(y_hist);
N = sum(y_hist*dt);

% 计算局部优化带宽


%窗口大小
WIN = logexp(linspace(ilogexp(max(5*dt)),ilogexp(1*T),M));

W = WIN;        %带宽

c = zeros(M,L);
for j = 1:M
    w = W(j);
    yh = fftkernel(y_hist,w/dt);
    %计算局部损失函数
    c(j,:) = yh.^2 - 2*yh.*y_hist + 2/sqrt(2*pi)/w*y_hist;
end


optws = zeros(M,L);
for i = 1:M
    Win = WIN(i);
    
    C_local = zeros(M,L);
    for j = 1:M
        C_local(j,:) = fftkernelWin(c(j,:),Win/dt,WinFunc);   
    end
    
    [mbuf,n] = min(C_local,[],1); 
    optws(i,:) = W(n);
end


% 使用黄金分割搜索方法搜索最佳带宽


% 参数初始化
tol = 10^-5; 
a = 1e-12; b = 1;

phi = (sqrt(5) + 1)/2;  %黄金分割率

c1 = (phi-1)*a + (2-phi)*b;
c2 = (2-phi)*a + (phi-1)*b;

f1 = CostFunction(y_hist,N,t,dt,optws,WIN,WinFunc,c1);
f2 = CostFunction(y_hist,N,t,dt,optws,WIN,WinFunc,c2);
    
k = 1;
while  ( abs(b-a) > tol*(abs(c1)+abs(c2)) ) && k < 30
    if f1 < f2
        b = c2;
        c2 = c1;
        c1 = (phi - 1)*a + (2 - phi)*b;
        
        f2 = f1;
        [f1 yv1 optwp1] = CostFunction(y_hist,N,t,dt,optws,WIN,WinFunc,c1);
        
        
        yopt = yv1 / sum(yv1*dt);
        optw = optwp1;
    else
        a = c1;
        c1 = c2;
        c2 = (2 - phi)*a + (phi - 1)*b;
        
        f1 = f2;
        [f2 yv2 optwp2] = CostFunction(y_hist,N,t,dt,optws,WIN,WinFunc,c2);
        
       
        yopt = yv2 / sum(yv2*dt);
        optw = optwp2;
    end
    
    gs(k) = (c1);
    C(k) = f1;
    k = k + 1;
end


% Bootstrap法求概率密度上下界
if nargout == 0 || nargout >= 6 || nargin >= 3

    yb = zeros(bootsamp_num,length(tin));

    for i = 1: bootsamp_num
        Nb = poissrnd(N);
   
        idx = ceil(rand(1,Nb)*N);
        xb = x_ab(idx);
        y_histb = histc(xb,t-dt/2);
        
        idx = find(y_histb ~= 0);
        y_histb_nz = y_histb(idx); 
        t_nz = t(idx);
        for k = 1: L
            yb_buf(k) = sum(y_histb_nz.*Gauss(t(k)-t_nz,optw(k)))/Nb;
        end
        yb_buf = yb_buf / sum(yb_buf*dt);
        
        yb(i,:) = interp1(t,yb_buf,tin);
        
    end

    ybsort = sort(yb);
    y95b = ybsort(floor(0.05*bootsamp_num),:);
    y95u = ybsort(floor(0.95*bootsamp_num),:); 
    
    confb95 = [y95b; y95u]; %区间上下界
    
end

%保存最优解
y = interp1(t,yopt,tin);
optw = interp1(t,optw,tin);
t = tin;



%损失函数
function [Cg yv optwp] = CostFunction(y_hist,N,t,dt,optws,WIN,WinFunc,g)

L = length(y_hist);
optwv = zeros(1,L);
for k = 1: L
	gs = optws(:,k)'./WIN;
        
	if g > max(gs) 
        optwv(k) = min(WIN);
    else
        if g < min(gs)
            optwv(k) = max(WIN);
        else
            idx = find(gs >= g, 1, 'last');
            optwv(k) = g*WIN(idx); 
          
        end
    end
end

%Nadaraya-Watson核回归
optwp = zeros(1,L);
for k = 1: L
    Z = feval(WinFunc,t(k)-t,optwv/g); 
    optwp(k) = sum(optwv.*Z)/sum(Z);
end
        
%使用优化带宽进行核密度估计


idx = find(y_hist ~= 0);
y_hist_nz = y_hist(idx); 
t_nz = t(idx);

yv = zeros(1,L);
for k = 1: L
    yv(k) = sum( y_hist_nz*dt.*Gauss(t(k)-t_nz,optwp(k)));    
end
yv = yv *N/sum(yv*dt); 



%核密度估计的损失函数
cg = yv.^2 - 2*yv.*y_hist + 2/sqrt(2*pi)./optwp.*y_hist;
Cg = sum(cg*dt);


function [y] = fftkernel(x,w)
L = length(x);
Lmax = L+3*w;  
 
n = 2^(ceil(log2(Lmax))); 

X = fft(x,n);

f = [-(0:n/2) (n/2-1:-1:1)]/n;

% Gauss
K = exp(-0.5*(w*2*pi*f).^2);


y = ifft(X.*K,n);
y = y(1:L);

function [y] = fftkernelWin(x,w,WinFunc)
L = length(x);
Lmax = L+3*w; 

n = 2^(ceil(log2(Lmax)));

X = fft(x,n);

f = [-(0:n/2) (n/2-1:-1:1)]/n;
t = 2*pi*f;

if strcmp(WinFunc,'Boxcar')
    %Boxcar
    a = sqrt(12)*w;
    K = 2*sin(a*t/2)./(a*t);
    K(1) = 1;
elseif strcmp(WinFunc,'Laplace')
    % Laplace
    K = 1 ./ ( 1+ (w*2*pi.*f).^2/2 );
elseif strcmp(WinFunc,'Cauchy')
    % Cauchy
    K = exp(-w*abs(2*pi*f));
elseif strcmp(WinFunc,'Gauss')
    % Gauss
    K = exp(-0.5*(w*2*pi*f).^2);
end

y = ifft(X.*K,n);
y = y(1:L);

function y = Gauss(x,w) 
y = 1/sqrt(2*pi)./w.*exp(-x.^2/2./w.^2);

function y = Laplace(x,w)
y = 1./sqrt(2)./w.*exp(-sqrt(2)./w.*abs(x));

function y = Cauchy(x,w) 
y = 1./(pi*w.*(1+ (x./w).^2));

function y = Boxcar(x,w)
a = sqrt(12)*w;
y = 1./a; y(abs(x) > a/2) = 0; 


function y = logexp(x)
idx = x<1e2;  
y(idx) = log(1+exp(x(idx)));

idx = x>=1e2; 
y(idx) = x(idx);

function y = ilogexp(x)
idx = x<1e2;  
y(idx) = log(exp(x(idx))-1);

idx = x>=1e2; 
y(idx) = x(idx);



