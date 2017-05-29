import numpy as np
import numpy.linalg as lin
import matplotlib.pyplot as plt
from sympy import *

x1= Symbol('x1')
x2= Symbol('x2')
X = np.array([[1.0],[x1],[x2]])
a = np.array([[0.],[0.],[0.]])

y = np.array([[1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0]
                 ,[0.1,6.8,-3.5,2.0,4.1,3.1,-0.8,0.9,5.0,3.9,-3.0,0.5,2.9,-0.1,-4.0,-1.3,-3.4,-4.1,-5.1,1.9]
                 ,[1.1,7.1,-4.1,2.7,2.8,5.0,-1.3,1.2,6.4,4.0,-2.9,8.7,2.1,5.2,2.2,3.7,6.2,3.4,1.6,5.1]])
b1 = 0.1
b2 = 0.5
Yk =[0]
learning_rate =0.47#0.47
count  = 0
while Yk.__len__() != 0:
    count = count +1
    Yk = []
    j = 0
    sum = np.array([[0],[0],[0]])

    while j != 20:
        if np.dot(a.transpose(),y[:,j]) <= b2:
            Yk.append(y[:,j])
        j = j + 1

    print(Yk.__len__())
    for i in np.arange(Yk.__len__()):
        sum = sum + ((b2- np.dot(a.transpose() , Yk[i].reshape(3,1)))/((lin.norm(Yk[i]))**2))*Yk[i].reshape(3,1)
    a = a + learning_rate*sum
    learning_rate = learning_rate - 0.0005
    if count == 200:
        break;
    print("a = ",a)

graph = np.dot(a.transpose(),X)
print(solve(graph[0]))
y1 = [0.964131542799681*x2 + 4.03482557075009  for x2 in np.linspace(-5,10,10)]
plt.plot(np.linspace(-5,10,10),y1)

for i in np.arange(20):
    if i <10:
        plt.scatter(y[1,i],y[2,i],marker="+",edgecolors='red',s=200)
    else :
        plt.scatter(y[1,i],y[2,i],marker="o",edgecolors='red',s=200)
plt.show()



