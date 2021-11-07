%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% ������
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function imt = JYJ_FCM(ima,Cluster,SD_Flag,Jyj_A,Jyj_XZ,Jyj_L,Jyj_DD,Jyj_P,Jyj_Q)
% ���趨FCM�ļ�����ʼ����
options=[Jyj_L;     % FCM��ʽ�еĲ��� ������Ȩ��ָ��L2
		 Jyj_DD;	% ����������100
		 1e-5];     % Ŀ�꺯������С���		
class_number = Cluster;  % ��Ϊ4��
imt = ImageSegmentation(ima,class_number,options,SD_Flag,Jyj_A,Jyj_XZ,Jyj_P,Jyj_Q);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% ImageSegmentation()������ʵ�־���ָ�ͼ��
% ���룺fileΪ�Ҷ�ͼ���ļ� cluster_nΪ���������� optionsΪԤ��ĳ�ʼ����
% ����ָ���ͼ��  
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function imt = ImageSegmentation(file, cluster_n, options,SD_Flag,Jyj_A,Jyj_XZ,Jyj_P,Jyj_Q) 
ima = file;
I = im2double(file);
%��1����ԭ��ά����ת��Ϊһά�о���
[x,y] = size(ima);
number = x * y;  % ͼ���Ԫ�ظ���numel(I)
data = reshape(I,number,1); %������Ԫ��ת��Ϊһ������

%��2�����þ��෽��
[center, U] = FCMprocess(data,cluster_n,options,y,x,SD_Flag,Jyj_A,Jyj_XZ,Jyj_P,Jyj_Q); %����FCMData�������о���

%��3������ÿ��Ԫ�ضԲ�ͬ�������ĵ������ȣ��ҳ������Ǹ�������
maxU = max(U); % �ҳ�ÿһ�е����������
temp = sort(center); 
for i = 1:cluster_n; % ���������ָ�ͼ��
    % ǰ�����ÿ��Ԫ�ص���������ȣ����ڸ��������ĵ�Ԫ�����꣬�������Щ����
    % ����eval��������������ַ���ת��Ϊ����ִ��
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
% ���ڼ���������ġ������Ⱦ����Ŀ�꺯��
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [center, U] = FCMprocess(data, cluster_num, options,IMG_W,IMG_H,SD_Flag,Jyj_A,Jyj_XZ,Jyj_P,Jyj_Q) 
%dataΪ��������,cluster_numΪ�����
L = options(1);                 % ����L
max_iteration = options(2);		% ���յĵ�������
min_deviation = options(3);		% ��С�б����
data_number = size(data, 1);    % Ԫ�ظ���
obj_function = zeros(max_iteration, 1); % obj_function���ڴ��Ŀ�꺯����ֵ
% ���������Ⱦ���U
U = rand(cluster_num, data_number); % ������������Ⱦ���U
sumU = sum(U,1);   % ����U��ÿ��Ԫ�غ�
for k = 1:data_number
    U(:,k) = U(:,k) ./ sumU(k);  % ����������U���й�һ������
end

for i = 1:max_iteration
	[U, center, obj_function(i)] = FCMStep(data, U, cluster_num, L,IMG_W,IMG_H,SD_Flag,Jyj_A,Jyj_XZ,Jyj_P,Jyj_Q); %����FCMStep�������е���
		fprintf('��%d�ε���, Ŀ�꺯��ֵΪ%f\n', i, obj_function(i));
	% ��������ֹ����
	if i > 1
        if abs(obj_function(i) - obj_function(i-1)) < min_deviation
            break;
        end
	end
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% �ú�������ÿ�ε�������
function [newU,Vm,J] = FCMStep(data, U, cluster_num, L,IMG_W,IMG_H,SD_Flag,Jyj_A,Jyj_XZ,Jyj_P,Jyj_Q)
% dataΪ����������,UΪ�����Ⱦ���,cluster_numΪ���������,mΪFCM�еĲ���m
% �������ú�õ��µ������Ⱦ���newU,��������center,Ŀ�꺯��ֵobj_function
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%��1�������Ǽ���ģ��������Ut
[x,y] = size(U);
A = ones(x,y);
a = Jyj_A;%0.85 0.8 0.7 0.5
Ut = abs(A - U -(A - (U).^a).^(1/a));
Ud = U  + Ut;

%��2������õ�obj
%[j,k,l] = size(data);
pp =  y;                    %������ÿ�е�Ԫ�ظ�����
pai = (sum(Ut,2)) ./pp;     %ÿ��������֮�� / ÿ��Ԫ�ظ��� == ÿ�е�ƽ��������
obj = sum(pai.*exp(1-pai)); %(ÿ��ƽ��������*exp(1-ÿ��ƽ��������)),����Ԫ����ӣ��õ�һ����ֵobj
%Ud = U;
%obj = 0;
if SD_Flag==0
%��3�����幫ʽ�е������ȳ�Ա����Umn �� Umn.^L ��L��������ģ�������ȵ�Ȩ��ָ��
Umn = Ud;               %ģ��������Umn[cluster_number * data_number]
Umn_L = Ud.^L;          % FMC�е�Umn.^L
else
%��3.5��Ϊ�����ȼӿռ�����Ϣ
Umn=SpatialDomain(Ud,IMG_W,IMG_H,Jyj_P,Jyj_Q);
Umn_L = Umn.^L;
end
%��4������õ���������Vm [4 1]�о���
%4.1 ���������data1
In = zeros(x,y);
%4.2 ��ԭ�о��� ת��Ϊ�о��� ����ֵ��data1��ÿһ�С�
for CI=1:cluster_num
   In(CI,:) = data';
end
%4.3 ����õ���������
if Jyj_XZ==0
    Vm = sum(Umn_L.*In,2)./sum(Umn_L,2);       % (ԭ��ʽЧ������)�õ���������
else
    Vm = sum(Umn.*In,2)./sum(Umn,2);            % �õ���������
end  
%��5�����㷶��|| In-Vm || ,�˴���llin_Vmll��ʾ
llin_Vmll = Distance(Vm, data);                 % ������������뱻�������ݵľ���

%��6�����ۺ���J
J = sum(sum((llin_Vmll.^2).*Umn_L))+obj;        % �õ�Ŀ�꺯��ֵ

%��7�������ȵ���
tmp = llin_Vmll.^(-2/(L-1));                    % �������������Ϊ1,�����µ������Ⱦ���
newU = tmp./(ones(cluster_num, 1)*sum(tmp));    % U_newΪ�µ������Ⱦ���

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Distance()�������ڼ�����������뱻�������ݵľ���
% centerΪ�������ģ�dataΪ���������ݣ������Ԫ�ص��������ĵľ���out
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function out = Distance(center, data)
  
 data_number = size(data,1);
 class_number = size(center, 1);
 kk = ones(data_number,1);  %���������ݴ�С��ͬ��ȫ1����kk
 out = zeros(class_number, data_number);
  if size(center, 2) > 1,   %�����������1
      for k = 1:class_number
  	     out(k, :) = sqrt(sum(((data - kk...
                       *center(k,:)).^2)'));
     end
  else	% dataΪһά����
      for k = 1:class_number
  	     out(k, :) = abs(center(k) - data)';
      end
  end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% ��ӿռ�����Ϣ
% 20190423
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function Umn_SD = SpatialDomain(Umn,IMG_W,IMG_H,Jyj_P,Jyj_Q)
%��1��C=clusternum,N=datanum
[C,N]=size(Umn);
%��2��һ�������е����ص�������ȣ�ת��Ϊ�ռ�����
Hmn = Umn;
jyj_start = IMG_W+2;%�ӵڶ��е�2�����ص㿪ʼ
jyj_end = IMG_W*(IMG_H-1)-1;%�������ڶ��е�����2�������
for i=jyj_start:jyj_end 
    if mod(i,IMG_W) == 1 || mod(i,IMG_W) == 0
        continue
    end
    Hmn(:,i) = Umn(:,i-1)+Umn(:,i+1)+Umn(:,i-IMG_W)+Umn(:,i-IMG_W-1)+Umn(:,i-IMG_W+1)+Umn(:,i+IMG_W)+Umn(:,i+IMG_W-1)+Umn(:,i+IMG_W+1);
end
%��3������
Umn_SD =(Hmn.^Jyj_Q) .* (Umn.^Jyj_P);
sumU = sum(Umn_SD,1);   % ����U��ÿ��Ԫ�غ�
for k = 1:N
    Umn_SD(:,k) = Umn_SD(:,k) ./ sumU(k);  % ����������U���й�һ������
end








  
