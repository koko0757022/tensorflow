# coding: utf-8

def ready(a,arr):  #클레스 별로 나눈다.
    for i in np.where(arr[:,4] == a):
        test = np.array([arr[i,0],arr[i,1],arr[i,2],arr[i,3]])

    return test

def ready_2(a,arr):  #클레스 별로 나눈다.
    for i in np.where(arr[:,4] == a):
        test = np.array([arr[i,0],arr[i,1]])

    return test


def class_mean(Class):
    result = np.array([[0],[0],[0],[0]],dtype=float)
    for i in np.arange(4):        #행
        sum = 0
        for j in np.arange(Class[i].__len__()):    #열
            sum += Class[i,j]
        result[i,0] = sum/Class[i].__len__()
    return result

def class_mean2(Class):
    result = np.array([[0],[0]],dtype=float)
    for i in np.arange(2):        #행
        sum = 0
        for j in np.arange(Class[i].__len__()):    #열

            sum += Class[i,j]
        result[i,0] = sum/Class[i].__len__()
    return result

def boundary():  ##3개의 클라스의 공분산이 달라서 case3을 이용했다
        i=-1  ##행을 늘리기 위한 counter

        V1 = (-1/2)*lin.inv(class1_cov)
        v1 = np.dot(lin.inv(class1_cov),class1_mean)
        v01 = (-1/2)*np.dot(np.dot(class1_mean.transpose(),lin.inv(class1_cov)),class1_mean)-(1/2)*np.log(lin.det(class1_cov))+np.log(0.33)

        V2 = (-1/2)*lin.inv(class2_cov)
        v2 = np.dot(lin.inv(class2_cov),class2_mean)
        v02 = (-1/2)*np.dot(np.dot(class2_mean.transpose(),lin.inv(class2_cov)),class2_mean)-(1/2)*np.log(lin.det(class2_cov))+np.log(0.33)

        V3 = (-1/2)*lin.inv(class3_cov)
        v3 = np.dot(lin.inv(class3_cov),class3_mean)
        v03 = (-1/2)*np.dot(np.dot(class3_mean.transpose(),lin.inv(class3_cov)),class3_mean)-(1/2)*np.log(lin.det(class3_cov))+np.log(0.33)

        for j in np.arange(30): ## test의 30개의 데이터개수
            #d1 ,d2 ,d3는 각 클래스에 해당하는 boundary 이다. ppt의 case3에 해당하는 공식을 이용했다.
            d1 = np.dot(np.dot(test[j,0:4],V1),test[j,0:4].transpose()) + np.dot(v1.transpose(),test[j,0:4].transpose())+v01
            d2 = np.dot(np.dot(test[j,0:4],V2),test[j,0:4].transpose()) + np.dot(v2.transpose(),test[j,0:4].transpose())+v02
            d3 = np.dot(np.dot(test[j,0:4],V3),test[j,0:4].transpose()) + np.dot(v3.transpose(),test[j,0:4].transpose())+v03

            if j%10 ==0:
                i = i+1
                # print("*****************class********",i+1)

            point = [d1[0],d2[0],d3[0]]
            # print("        d1           d2            d3\n",d1[0],d2[0],d3[0])
            compare(i,point)


def compare(a,point):  ##각 클래스의 특징값을 boundary 에 넣어서  가장 큰  boundary쪽에 count++ 을 하도록함
    j=0
    max = point[0]
    for i in np.arange(3):
        if max < point[i]:
            max = point[i]
            j=i
    count[a][j] = count[a][j] + 1


def boundary_21():
        i=-1  ##행을 늘리기 위한 counter

        V1 = (-1/2)*lin.inv(class1_2feature_cov)
        v1 = np.dot(lin.inv(class1_2feature_cov),class1_2feature_mean)
        v01 = (-1/2)*np.dot(np.dot(class1_2feature_mean.transpose(),lin.inv(class1_2feature_cov)),class1_2feature_mean)-(1/2)*np.log(lin.det(class1_2feature_cov))+np.log(0.33)

        V2 = (-1/2)*lin.inv(class2_2feature_cov)
        v2 = np.dot(lin.inv(class2_2feature_cov),class2_2feature_mean)
        v02 = (-1/2)*np.dot(np.dot(class2_2feature_mean.transpose(),lin.inv(class2_2feature_cov)),class2_2feature_mean)-(1/2)*np.log(lin.det(class2_2feature_cov))+np.log(0.33)

        V3 = (-1/2)*lin.inv(class3_2feature_cov)
        v3 = np.dot(lin.inv(class3_2feature_cov),class3_2feature_mean)
        v03 = (-1/2)*np.dot(np.dot(class3_2feature_mean.transpose(),lin.inv(class3_2feature_cov)),class3_2feature_mean)-(1/2)*np.log(lin.det(class3_2feature_cov))+np.log(0.33)

        d1_p = np.dot(np.dot(x1_x2,V1),x1_x2.transpose()) + np.dot(v1.transpose(),x1_x2.transpose())+v01
        d2_p = np.dot(np.dot(x1_x2,V2),x1_x2.transpose()) + np.dot(v2.transpose(),x1_x2.transpose())+v02
        d3_p = np.dot(np.dot(x1_x2,V3),x1_x2.transpose()) + np.dot(v3.transpose(),x1_x2.transpose())+v03

        d1_p = d1_p[0,0]    # 각 d의 전개식을 표현
        d2_p = d2_p[0,0]
        d3_p = d3_p[0,0]
        #
        d1_p_expand = (d1_p).expand()   #expand는 전개식을 풀어주는 함수 이다.
        d2_p_expand = (d2_p).expand()
        d3_p_expand = (d3_p).expand()
        #
        expr1 = d1_p_expand - d2_p_expand
        expr2 = d1_p_expand - d3_p_expand
        expr3 = d2_p_expand - d3_p_expand


        # print(solve(expr1))
        # print(solve(expr2))
        # print(solve(expr3))

        #각 expr의 x1의 해가나오는데 그게 두개씩이다
        expr1_1=0.601793500659381*x2 - 9.01451820931711e-16*sqrt(2.70624865454661e+29*x2**2 - 1.67507271942675e+29*x2 - 9.49159371731281e+28) + 2.22023281344768
        expr1_2=0.601793500659381*x2 + 9.01451820931711e-16*sqrt(2.70624865454661e+29*x2**2 - 1.67507271942675e+29*x2 - 9.49159371731281e+28) + 2.22023281344768

        expr2_1=0.64176711287483*x2 - 7.39268927173037e-16*sqrt(5.82420054173064e+29*x2**2 - 2.32118070816799e+30*x2 + 3.3977356644497e+30) + 2.39623676812231
        expr2_2=0.64176711287483*x2 + 7.39268927173037e-16*sqrt(5.82420054173064e+29*x2**2 - 2.32118070816799e+30*x2 + 3.3977356644497e+30) + 2.39623676812231

        expr3_1=0.823976523083762*x2 - 4.10903582439455e-15*sqrt(4.7812229599438e+28*x2**2 - 3.59747847024408e+29*x2 + 6.78603325781985e+29) + 3.19850543983648
        expr3_2=0.823976523083762*x2 + 4.10903582439455e-15*sqrt(4.7812229599438e+28*x2**2 - 3.59747847024408e+29*x2 + 6.78603325781985e+29) + 3.19850543983648


        x = np.linspace(2.0,4.5,100)
        y = [0.601793500659381*x2 - 9.01451820931711e-16*sqrt(2.70624865454661e+29*x2**2 - 1.67507271942675e+29*x2 - 9.49159371731281e+28) + 2.22023281344768 for x2 in x]
        plt.plot(x,y,c="blue")
        y1 = [0.601793500659381*x2 + 9.01451820931711e-16*sqrt(2.70624865454661e+29*x2**2 - 1.67507271942675e+29*x2 - 9.49159371731281e+28) + 2.22023281344768 for x2 in x]
        plt.plot(x,y1,c="blue")

        y = [0.823976523083762*x2 - 4.10903582439455e-15*sqrt(4.7812229599438e+28*x2**2 - 3.59747847024408e+29*x2 + 6.78603325781985e+29) + 3.19850543983648 for x2 in x]
        plt.plot(x,y,c="yellow")
        y1 = [0.823976523083762*x2 + 4.10903582439455e-15*sqrt(4.7812229599438e+28*x2**2 - 3.59747847024408e+29*x2 + 6.78603325781985e+29) + 3.19850543983648 for x2 in x]
        plt.plot(x,y1,c="yellow")

        y = [0.64176711287483*x2 - 7.39268927173037e-16*sqrt(5.82420054173064e+29*x2**2 - 2.32118070816799e+30*x2 + 3.3977356644497e+30) + 2.39623676812231 for x2 in x]
        plt.plot(x,y,c="black")
        y1 = [0.64176711287483*x2 + 7.39268927173037e-16*sqrt(5.82420054173064e+29*x2**2 - 2.32118070816799e+30*x2 + 3.3977356644497e+30) + 2.39623676812231 for x2 in x]
        plt.plot(x,y1,c="black")



        for j in np.arange(30): ## test의 30개의 데이터개수

             #d1 ,d2 ,d3는 각 클래스에 해당하는 boundary 이다.

            d1 = np.dot(np.dot(test[j,0:2],V1),test[j,0:2].transpose()) + np.dot(v1.transpose(),test[j,0:2].transpose())+v01
            d2 = np.dot(np.dot(test[j,0:2],V2),test[j,0:2].transpose()) + np.dot(v2.transpose(),test[j,0:2].transpose())+v02
            d3 = np.dot(np.dot(test[j,0:2],V3),test[j,0:2].transpose()) + np.dot(v3.transpose(),test[j,0:2].transpose())+v03

            if j%10 ==0:
                i = i+1
                # print("*****************class********",i+1)

            point = [d1[0],d2[0],d3[0]]
            # print("        d1           d2            d3\n",d1[0],d2[0],d3[0])
            compare(i,point)

def compare(a,point):  ##각 클래스의 특징값을 boundary 에 넣어서  가장 큰  boundary쪽에 count++ 을 하도록함
    j=0
    max = point[0]
    for i in np.arange(3):
        if max < point[i]:
            max = point[i]
            j=i
    count[a][j] = count[a][j] + 1

def contour2():

    x1 = Symbol('x1')
    x2 = Symbol('x2')
    x1_x2 = np.array([x1,x2])

    u1 = class1_2feature_mean.transpose()
    v1 = x1_x2
    V1 = class1_2feature_cov
    VI1 = lin.inv(V1)

    u2 = class2_2feature_mean.transpose()
    v2 = x1_x2
    V2 = class2_2feature_cov
    VI2 = lin.inv(V2)

    u3 = class3_2feature_mean.transpose()
    v3 = x1_x2
    V3 = class3_2feature_cov
    VI3 = lin.inv(V3)

    class1=(sqrt((np.dot(np.dot((v1-u1),VI1),(v1-u1).transpose()))[0,0])-2).expand() ##  = 2니깐 넘겨서 -2로 함
    class2=(sqrt((np.dot(np.dot((v2-u2),VI2),(v2-u2).transpose()))[0,0])-2).expand()
    class3=(sqrt((np.dot(np.dot((v3-u3),VI3),(v3-u3).transpose()))[0,0])-2).expand()

    # a = solve(class1)   #Mahalanobis 의 해를 구하기위해서 사용
    # print(a)
    # a = solve(class2)
    # print(a)
    # a = solve(class3)
    # print(a)
    #
    # class1_1 = 0.685178374402354*x2 - 6.14782961307419e-15*sqrt(-1.03693386632142e+28*x2**2 + 7.08744297630698e+28*x2 - 1.17853501262421e+29) + 2.65590290547996
    # class1_2 = 0.685178374402354*x2 + 6.14782961307419e-15*sqrt(-1.03693386632142e+28*x2**2 + 7.08744297630698e+28*x2 - 1.17853501262421e+29) + 2.65590290547996
    # class2_1 = 0.86400351667674*x2 - 9.66615663582529e-16*sqrt(-1.7735900374614e+30*x2**2 + 9.85229265809812e+30*x2 - 1.32685564918599e+31) + 3.59023023243035
    # class2_2 = 0.86400351667674*x2 + 9.66615663582529e-16*sqrt(-1.7735900374614e+30*x2**2 + 9.85229265809812e+30*x2 - 1.32685564918599e+31) + 3.59023023243035
    # class3_1 = 0.899568034557236*x2 - 3.65093315611674e-15*sqrt(-2.88396341898128e+29*x2**2 + 1.71307427087489e+30*x2 - 2.48913477684457e+30) + 3.93828293736502
    # class3_2 = 0.899568034557236*x2 + 3.65093315611674e-15*sqrt(-2.88396341898128e+29*x2**2 + 1.71307427087489e+30*x2 - 2.48913477684457e+30) + 3.93828293736502

    class1_1=0.685178374402354*x2 - 1.53695740326855e-17*sqrt(-1.65909418611427e+33*pow(x2,2) + 1.13399087620911e+34*x2 - 1.83360513067514e+34) + 2.65590290547996
    class1_2=0.685178374402354*x2 + 1.53695740326855e-17*sqrt(-1.65909418611427e+33*pow(x2,2) + 1.13399087620911e+34*x2 - 1.83360513067514e+34) + 2.65590290547996
    class2_1=0.86400351667674*x2 - 4.83307831791264e-17*sqrt(-7.09436014984561e+32*pow(x2,2) + 3.94091706323924e+33*x2 - 5.14189662191442e+33) + 3.59023023243035
    class2_2=0.86400351667674*x2 + 4.83307831791264e-17*sqrt(-7.09436014984561e+32*pow(x2,2) + 3.94091706323924e+33*x2 - 5.14189662191442e+33) + 3.59023023243035
    class3_1=0.899568034557236*x2 - 1.82546657805837e-17*sqrt(-1.15358536759251e+34*pow(x2,2) + 6.85229708349953e+34*x2 - 9.73741704575963e+34) + 3.93828293736502
    class3_2=0.899568034557236*x2 + 1.82546657805837e-17*sqrt(-1.15358536759251e+34*pow(x2,2) + 6.85229708349953e+34*x2 - 9.73741704575963e+34) + 3.93828293736502
    x3 = np.linspace(2.63,4.2,1000)
    class1_1 = [0.685178374402354*x2 - 1.53695740326855e-17*sqrt(-1.65909418611427e+33*pow(x2,2) + 1.13399087620911e+34*x2 - 1.83360513067514e+34) + 2.65590290547996 for x2 in x3]

    class1_2=  [0.685178374402354*x2 + 1.53695740326855e-17*sqrt(-1.65909418611427e+33*pow(x2,2) + 1.13399087620911e+34*x2 - 1.83360513067514e+34) + 2.65590290547996 for x2 in x3]
    plt.plot(x3,class1_1,c='green')
    plt.plot(x3,class1_2,c='green')

    x3 = np.linspace(2.1,3.45,1000)
    class2_1 = [0.86400351667674*x2 - 4.83307831791264e-17*sqrt(-7.09436014984561e+32*pow(x2,2) + 3.94091706323924e+33*x2 - 5.14189662191442e+33) + 3.59023023243035 for x2 in x3]

    class2_2 = [0.86400351667674*x2 + 4.83307831791264e-17*sqrt(-7.09436014984561e+32*pow(x2,2) + 3.94091706323924e+33*x2 - 5.14189662191442e+33) + 3.59023023243035 for x2 in x3]
    plt.plot(x3,class2_1,c='green')
    plt.plot(x3,class2_2,c='green')

    x3 = np.linspace(2.357,3.585,1000)
    class3_1 = [0.899568034557236*x2 - 1.82546657805837e-17*sqrt(-1.15358536759251e+34*pow(x2,2) + 6.85229708349953e+34*x2 - 9.73741704575963e+34) + 3.93828293736502 for x2 in x3]

    class3_2 = [0.899568034557236*x2 + 1.82546657805837e-17*sqrt(-1.15358536759251e+34*pow(x2,2) + 6.85229708349953e+34*x2 - 9.73741704575963e+34) + 3.93828293736502 for x2 in x3]
    plt.plot(x3,class3_1,c='green')
    plt.plot(x3,class3_2,c='green')

    plt.scatter(class1_2feature_mean[1,0],class1_2feature_mean[0,0],s=200,marker='^',color='green')
    plt.scatter(class2_2feature_mean[1,0],class2_2feature_mean[0,0],s=200,marker='^',color='green')
    plt.scatter(class3_2feature_mean[1,0],class3_2feature_mean[0,0],s=200,marker='^',color='green')


import numpy as np
import numpy.linalg as lin
import matplotlib.pyplot as plt
from sympy import *

count = np.array([[0,0,0],[0,0,0],[0,0,0]])  #confusion matrix를 위한 초기화

traning = np.loadtxt("training.txt")
test = np.loadtxt("test.txt")

#ready() 함수는 traning 데이터를 각 클래스 별로 나누기 위한 함수이다 ready(클래스명,나눌데이터)
class1 = ready(1,traning) ##training
class2 = ready(2,traning)
class3 = ready(3,traning)


class1_mean = class_mean(class1) #class1의 mean벡터
class2_mean = class_mean(class2) #class2의 mean벡터
class3_mean = class_mean(class3) #class3의 mean벡터

mean_array = np.concatenate((class1_mean,class2_mean,class3_mean), axis=1) #각 클래스의 mean벡터를 하나로 합친다.

# print("각 클래스 의 mean 벡터\n",mean_array)

class1_cov = np.cov(class1)  #class1의 공분산
# print("class1 의 covariance\n",class1_cov)
class2_cov = np.cov(class2) #class2의 공분산
# print("class2 의 covariance\n",class2_cov)
class3_cov = np.cov(class3)  #class3의 공분산
# print("class3 의 covariance\n",class3_cov)

boundary()  ## test 데이터를 가지고 바운더리를 결정하여 3x3의 행렬로 confusion matrix에 카운트를 부과한다
# print("confusion matrix\n",count) ## confusion matrix의 값을 출력한다.

################################## 2번 문제###################
count = np.array([[0,0,0],[0,0,0],[0,0,0]])
x1 = Symbol('x1',dtype=float)
x2 = Symbol('x2',dtype=float)
x1_x2 = np.array([x1,x2])


plt.scatter(class1[1,:],class1[0,:],marker="+",edgecolors='red',s=200)  # 첫번째 클래스의 1, 2, 특징을 화면상에 나타낸다
plt.scatter(class2[1,:],class2[0,:],marker="o",edgecolors='gray',s=200)  # 두번째 클래스의 1, 2, 특징을 화면상에 나타낸다
plt.scatter(class3[1,:],class3[0,:],marker=".",edgecolors='black',s=200)  # 세번째 클래스의 1, 2, 특징을 화면상에 나타낸다


class1_2feature_mean = class_mean(class1)[0:2,:] #class1의 mean벡터
class2_2feature_mean = class_mean(class2)[0:2,:] #class2의 mean벡터
class3_2feature_mean = class_mean(class3)[0:2,:] #class3의 mean벡터
mean_array_2feature=mean_array[0:2,:]
# print("2개특징을 가지는 각 클래스의 mean 벡터 \n",mean_array_2feature)

class1_2feature = class1[0:2,:] # 각 클래스에서 두개의 특징만을 뽑아냄
class2_2feature = class2[0:2,:]
class3_2feature = class3[0:2,:]

class1_2feature_cov = np.cov(class1_2feature)  #두개의 특징을 뽑아낸 클래스의 covariance
class2_2feature_cov = np.cov(class2_2feature)
class3_2feature_cov = np.cov(class3_2feature)

# print("class1 의 2개특징의 covariance\n",class1_2feature_cov)
# print("class2 의 2개특징의 covariance\n",class2_2feature_cov)
# print("class3 의 2개특징의 covariance\n",class3_2feature_cov)

boundary_21()  #boundary 시행
print("confusion matirx\n",count.transpose())  ##confusion matrix 출력 test
plt.show()
##########contour############

contour2()  ##윤곽선 그리기

plt.show()
plt.scatter(test[0:10,1],test[0:10,0],marker="+",edgecolors='red',s=200)
plt.scatter(test[10:20,1],test[10:20,0],marker="o",edgecolors='gray',s=200)
plt.scatter(test[20:30,1],test[20:30,0],marker=".",edgecolors='black',s=200)

boundary_21()
plt.show()


