import numpy as np
import numpy.linalg as lin
import matplotlib.pyplot as plt
from sympy import *

y = np.array([[1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0]
                 ,[0.1,6.8,-3.5,2.0,4.1,3.1,-0.8,0.9,5.0,3.9,7.1,-1.4,4.5,6.3,4.2,1.4,2.4,2.5,8.4,4.1]
                 ,[1.1,7.1,-4.1,2.7,2.8,5.0,-1.3,1.2,6.4,4.0,4.2,-4.3,0.0,1.6,1.9,-3.2,-4.0,-6.1,3.7,-2.2]])

x1= Symbol('x1')
x2= Symbol('x2')
X = np.array([[1.0],[x1],[x2]])
a= np.array([[0.7],[0.8],[0.7]]) #0.02 0.2 0.2
learning_rate = 0.2#0.02
eta = np.array([[0.002],[0.002],[0.002]])
i = 0
sum = np.array([[0],[0],[0]])
Yk = []

# while i != 20:
#
#         if np.dot(a.transpose(),y[:,i])[0] < 0:  #aTy < 0
#            Yk.append(y[:,i])
#         # print(Yk)
#         i = i + 1
# i = 0
count = 0
while  True:

    k = 0
    Yk = []
    sum = np.array([[0],[0],[0]])
    while k != 20:
        if np.dot(a.transpose(),y[:,k]) <= 0:  #aTy < 0
            Yk.append(y[:,k])
        k = k + 1

    for i in np.arange(Yk.__len__()):
        sum = sum + Yk[i].reshape(3,1)
    leaning_rate = learning_rate -0.0001
    a = a + learning_rate * sum

    count = count + 1
    print(count)
    if (learning_rate * sum)[0] < eta[0] and (learning_rate * sum)[1] < eta[1] and (learning_rate * sum)[2] < eta[2]:
        break;

graph = np.dot(a.transpose(),X)
print(solve(graph[0]))
y1 = [0.621739130434783*x2 - 2.08695652173913 for x2 in np.linspace(-5,10,10)]
plt.plot(np.linspace(-5,10,10),y1)

for i in np.arange(20):
    if i <10:
        plt.scatter(y[1,i],y[2,i],marker="+",edgecolors='red',s=200)
    else :
        plt.scatter(y[1,i],y[2,i],marker="o",edgecolors='red',s=200)
plt.show()
