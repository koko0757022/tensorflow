import numpy as np
import numpy.linalg as lin
import matplotlib.pyplot as plt
from sympy import *


y = np.array([[1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0]
                 ,[0.1,6.8,-3.5,2.0,4.1,3.1,-0.8,0.9,5.0,3.9,-3.0,0.5,2.9,-0.1,-4.0,-1.3,-3.4,-4.1,-5.1,1.9]
                 ,[1.1,7.1,-4.1,2.7,2.8,5.0,-1.3,1.2,6.4,4.0,-2.9,8.7,2.1,5.2,2.2,3.7,6.2,3.4,1.6,5.1]])

x1= Symbol('x1')
x2= Symbol('x2')
X = np.array([[1.0],[x1],[x2]])
a = np.array([[0.3],[-0.2],[-0.2]])
#a = np.array([[0.3],[0.2],[-0.2]])
#a = np.array([[0.3],[-0.2],[0.2]])

k = 0
b = 0.5
Yk =[0]
learning_rate =0.047#0.47
eta = np.array([[0.002],[0.002],[0.002]])
count  = 0
k = 0
while True:

    Yk = []
    if np.dot(a.transpose(),y[:,k]) <= b:
        Yk.append(y[:,k])
        a = a + learning_rate*(b - np.dot(a.transpose(),Yk[0].reshape(3,1)))*Yk[0].reshape(3,1)
        print(a)
        d = abs(learning_rate*(b - np.dot(a.transpose(),Yk[0].reshape(3,1)))*Yk[0].reshape(3,1))
        print("d[0] =",d[0])
        print("d[1] =",d[1])
        print("d[2] =",d[2])
        if d[0] < eta[0] and d[1] < eta[1] and d[2] < eta[2]:
            break;
    k = k + 1

    if k == 20:
        k=0


    count = count +1
    if count == 100:
        break;


graph = np.dot(a.transpose(),X)
print(solve(graph[0]))
y1 = [-1.28597612931001*x2 - 23.0921967852023 for x2 in np.linspace(-5,10,10)]
plt.plot(np.linspace(-5,10,10),y1)

for i in np.arange(20):
    if i <10:
        plt.scatter(y[1,i],y[2,i],marker="+",edgecolors='red',s=200)
    else :
        plt.scatter(y[1,i],y[2,i],marker="o",edgecolors='red',s=200)
plt.show()



