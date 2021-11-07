%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 主函数
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function imt = JYJ_FCM(ima,Cluster,SD_Flag,Jyj_A,Jyj_XZ,Jyj_L,Jyj_DD,Jyj_P,Jyj_Q)
% 先设定FCM的几个初始参数
options=[Jyj_L;     % FCM公式中的参数 隶属度权重指数L2
		 Jyj_DD;	% 最大迭代次数100
		 1e-5];     % 目标函数的最小误差		
class_number = Cluster;  % 分为4类
imt = ImageSegmentation(ima,class_number,options,SD_Flag,Jyj_A,Jyj_XZ,Jyj_P,Jyj_Q);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% ImageSegmentation()函数：实现聚类分割图像
% 输入：file为灰度图像文件 cluster_n为聚类类别个数 options为预设的初始参数
% 输出分割后的图像  
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function imt = ImageSegmentation(file, cluster_n, options,SD_Flag,Jyj_A,Jyj_XZ,Jyj_P,Jyj_Q) 
ima = file;
I = im2double(file);
%（1）将原二维矩阵转换为一维列矩阵
[x,y] = size(ima);
number = x * y;  % 图像的元素个数numel(I)
data = reshape(I,number,1); %将矩阵元素转换为一列数据

%（2）调用聚类方法
[center, U] = FCMprocess(data,cluster_n,options,y,x,SD_Flag,Jyj_A,Jyj_XZ,Jyj_P,Jyj_Q); %调用FCMData函数进行聚类

%（3）对于每个元素对不同聚类中心的隶属度，找出最大的那个隶属度
maxU = max(U); % 找出每一列的最大隶属度
temp = sort(center); 
for i = 1:cluster_n; % 按聚类结果分割图像
    % 前面求出每个元素的最大隶属度，属于各聚类中心的元素坐标，并存放这些坐标
    % 调用eval函数将括号里的字符串转化为命令执行
    eval(['class_',int2str(i), '= find(U(', int2str(i), ',:) == maxU);']); 
    %gray = round(255 * (i-1) / (cluster_n-1));
        index = find(temp == center(i)); 
    switch index 
        case 1 
            gray = 0; 
        case cluster_n 
            gray = 255; 
        otherwise 
            gray = fix(255*(index-1)/(cluster_n-1)); 
    end 
    eval(['I(class_',int2str(i), '(:))=', int2str(gray),';']); 
end; 
imt = mat2gray(I); 

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 用于计算聚类中心、隶属度矩阵和目标函数
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [center, U] = FCMprocess(data, cluster_num, options,IMG_W,IMG_H,SD_Flag,Jyj_A,Jyj_XZ,Jyj_P,Jyj_Q) 
%data为聚类数据,cluster_num为类别数
L = options(1);                 % 参数L
max_iteration = options(2);		% 最终的迭代次数
min_deviation = options(3);		% 最小判别误差
data_number = size(data, 1);    % 元素个数
obj_function = zeros(max_iteration, 1); % obj_function用于存放目标函数的值
% 生成隶属度矩阵U
U = rand(cluster_num, data_number); % 随机生成隶属度矩阵U
sumU = sum(U,1);   % 计算U中每列元素和
for k = 1:data_number
    U(:,k) = U(:,k) ./ sumU(k);  % 对隶属矩阵U进行归一化处理
end

for i = 1:max_iteration
	[U, center, obj_function(i)] = FCMStep(data, U, cluster_num, L,IMG_W,IMG_H,SD_Flag,Jyj_A,Jyj_XZ,Jyj_P,Jyj_Q); %调用FCMStep函数进行迭代
		fprintf('第%d次迭代, 目标函数值为%f\n', i, obj_function(i));
	% 检查迭代终止条件
	if i > 1
        if abs(obj_function(i) - obj_function(i-1)) < min_deviation
            break;
        end
	end
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 该函数用于每次迭代过程
function [newU,Vm,J] = FCMStep(data, U, cluster_num, L,IMG_W,IMG_H,SD_Flag,Jyj_A,Jyj_XZ,Jyj_P,Jyj_Q)
% data为被聚类数据,U为隶属度矩阵,cluster_num为聚类类别数,m为FCM中的参数m
% 函数调用后得到新的隶属度矩阵newU,聚类中心center,目标函数值obj_function
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%（1）以下是计算模糊隶属度Ut
[x,y] = size(U);
A = ones(x,y);
a = Jyj_A;%0.85 0.8 0.7 0.5
Ut = abs(A - U -(A - (U).^a).^(1/a));
Ud = U  + Ut;

%（2）计算得到obj
%[j,k,l] = size(data);
pp =  y;                    %列数（每行的元素个数）
pai = (sum(Ut,2)) ./pp;     %每行隶属度之和 / 每行元素个数 == 每行的平均隶属度
obj = sum(pai.*exp(1-pai)); %(每行平均隶属度*exp(1-每行平均隶属度)),将列元素相加，得到一个数值obj
%Ud = U;
%obj = 0;
if SD_Flag==0
%（3）定义公式中的隶属度成员函数Umn 和 Umn.^L ，L是作用于模糊隶属度的权重指数
Umn = Ud;               %模糊隶属度Umn[cluster_number * data_number]
Umn_L = Ud.^L;          % FMC中的Umn.^L
else
%（3.5）为隶属度加空间域信息
Umn=SpatialDomain(Ud,IMG_W,IMG_H,Jyj_P,Jyj_Q);
Umn_L = Umn.^L;
end
%（4）计算得到聚类中心Vm [4 1]列矩阵
%4.1 创建零矩阵data1
In = zeros(x,y);
%4.2 将原列矩阵 转置为行矩阵 ，赋值给data1的每一行。
for CI=1:cluster_num
   In(CI,:) = data';
end
%4.3 计算得到聚类中心
if Jyj_XZ==0
    Vm = sum(Umn_L.*In,2)./sum(Umn_L,2);       % (原公式效果不好)得到聚类中心
else
    Vm = sum(Umn.*In,2)./sum(Umn,2);            % 得到聚类中心
end  
%（5）计算范数|| In-Vm || ,此处用llin_Vmll表示
llin_Vmll = Distance(Vm, data);                 % 计算聚类中心与被聚类数据的距离

%（6）代价函数J
J = sum(sum((llin_Vmll.^2).*Umn_L))+obj;        % 得到目标函数值

%（7）隶属度迭代
tmp = llin_Vmll.^(-2/(L-1));                    % 如果迭代次数不为1,计算新的隶属度矩阵
newU = tmp./(ones(cluster_num, 1)*sum(tmp));    % U_new为新的隶属度矩阵

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Distance()函数用于计算聚类中心与被聚类数据的距离
% center为聚类中心，data为被聚类数据，输出各元素到聚类中心的距离out
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function out = Distance(center, data)
  
 data_number = size(data,1);
 class_number = size(center, 1);
 kk = ones(data_number,1);  %构造与数据大小相同的全1矩阵kk
 out = zeros(class_number, data_number);
  if size(center, 2) > 1,   %若类别数大于1
      for k = 1:class_number
  	     out(k, :) = sqrt(sum(((data - kk...
                       *center(k,:)).^2)'));
     end
  else	% data为一维数据
      for k = 1:class_number
  	     out(k, :) = abs(center(k) - data)';
      end
  end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 添加空间域信息
% 20190423
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function Umn_SD = SpatialDomain(Umn,IMG_W,IMG_H,Jyj_P,Jyj_Q)
%（1）C=clusternum,N=datanum
[C,N]=size(Umn);
%（2）一行是所有的像素点的隶属度，转换为空间域函数
Hmn = Umn;
jyj_start = IMG_W+2;%从第二行第2个像素点开始
jyj_end = IMG_W*(IMG_H-1)-1;%到倒数第二行倒数第2个点结束
for i=jyj_start:jyj_end 
    if mod(i,IMG_W) == 1 || mod(i,IMG_W) == 0
        continue
    end
    Hmn(:,i) = Umn(:,i-1)+Umn(:,i+1)+Umn(:,i-IMG_W)+Umn(:,i-IMG_W-1)+Umn(:,i-IMG_W+1)+Umn(:,i+IMG_W)+Umn(:,i+IMG_W-1)+Umn(:,i+IMG_W+1);
end
%（3）计算
Umn_SD =(Hmn.^Jyj_Q) .* (Umn.^Jyj_P);
sumU = sum(Umn_SD,1);   % 计算U中每列元素和
for k = 1:N
    Umn_SD(:,k) = Umn_SD(:,k) ./ sumU(k);  % 对隶属矩阵U进行归一化处理
end








  
