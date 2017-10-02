% 多行注释ctrl+r 取消 ctrl+t
% 问题来源：http://blog.csdn.net/on2way/article/details/47729419
% 用matlab绘制svm中拉格朗日乘子问题

[x1,x2]=meshgrid(-10:0.1:10);
f=x1.^2-2*x1+1+x2.^2+4*x2+4 ; %f 函数 ，注意乘方要用 .^ 
g1=10-x1-10*x2;  % g1 g2 不等式约束条件
g2=10*x1-x2-10;

a1=58/101;   % a1,a2 分别是 g1 g2的拉格朗日乘子，这里用最优解的结果
a2=4/101;

g1=a1*g1;   
g2=a2*g2;

% 绘制曲面图形
mesh(x1,x2,f)
hold on
mesh(x1,x2,g1)
mesh(x1,x2,g2)
surf(x1,x2,f)      %让f曲面 更加明显

title('绘制svm中拉格朗日乘子问题');    
xlabel('X轴');        
ylabel('Y轴');       
zlabel('Z轴');         
