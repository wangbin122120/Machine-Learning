% ����ע��ctrl+r ȡ�� ctrl+t
% ������Դ��http://blog.csdn.net/on2way/article/details/47729419
% ��matlab����svm���������ճ�������

[x1,x2]=meshgrid(-10:0.1:10);
f=x1.^2-2*x1+1+x2.^2+4*x2+4 ; %f ���� ��ע��˷�Ҫ�� .^ 
g1=10-x1-10*x2;  % g1 g2 ����ʽԼ������
g2=10*x1-x2-10;

a1=58/101;   % a1,a2 �ֱ��� g1 g2���������ճ��ӣ����������Ž�Ľ��
a2=4/101;

g1=a1*g1;   
g2=a2*g2;

% ��������ͼ��
mesh(x1,x2,f)
hold on
mesh(x1,x2,g1)
mesh(x1,x2,g2)
surf(x1,x2,f)      %��f���� ��������

title('����svm���������ճ�������');    
xlabel('X��');        
ylabel('Y��');       
zlabel('Z��');         
