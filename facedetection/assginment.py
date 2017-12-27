import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import scipy.misc
import scipy.io
import random
import cv2
from skimage import color
from skimage import io
import glob
import numpy as np
import math
import copy
import cv2
from numpy import linalg as lin
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image

train_images = []
test_images= []

for image_path in glob.glob("./pos_train/*.png"):
    gray_image = cv2.imread(image_path,0)
    train_images.append(gray_image)

train_images = np.array(train_images).reshape(700,134*70)
train_labels = np.zeros((700,2))   #
for i in range(700):
    train_labels[i][1] = 1

#*************************************test_image****************#
for image_path in glob.glob("./pos_test/*.png"):                                        #192개
    gray_image = cv2.imread(image_path,0)
    test_images.append(gray_image)
test_images = np.array(test_images).reshape(192,134*70)

test_labels = np.zeros((192,2))   # 700개 image중에서 350개의 class
for i in range(192):
    test_labels[i][1]=1

train_images = train_images/255.   #정규화를 시킨다.
test_images = test_images/255.

size = 70,134
neg_train = []
neg_test = []

for infile in glob.glob("./neg_train/*.bmp"):   ##neg_traing_image   ##500,134,70
    image = cv2.imread(infile,0)
    resize = cv2.resize(image, size)
    neg_train.append(resize)
neg_train = np.array(neg_train).reshape(500,134*70)

for infile in glob.glob("./neg_test/*.bmp"):  ## neg_test_image  ##99 , 134 , 70
    print(infile)
    image = cv2.imread(infile,0)
    resize = cv2.resize(image, size)

    neg_test.append(resize)
neg_test = np.array(neg_test).reshape(99,134*70)

neg_train_labels = np.zeros((500,2))   # 500개 neg_train의 labels
for i in range(500):
    neg_train_labels[i][0]=1

neg_test_labels = np.zeros((99,2))   # 500개 neg_test의 labels
for i in range(99):
    neg_test_labels[i][0]=1


neg_train = neg_train/255.   #정규화를 시킨다.
neg_test = neg_test/255.


##한곳에 몰아 넣기
test_labels = np.concatenate((neg_test_labels,test_labels))
train_labels = np.concatenate((neg_train_labels,train_labels))

train_images = np.concatenate((neg_train,train_images))
test_images = np.concatenate((neg_test,test_images))

###########################cropping#################################3
store=[]  ## crop된 이미지가 저장될 저장소
def cropping(img,multiple):
    global row,col
    num = 0
    img = pyramid(img, multiple)  #pyramid함수를 통해 어떤이미지를 몇배율로 할지 정한다.
    crop_img = img[0:134, 0:70] #70 * 134 의size로 원본 이미지를 crop한다.
    row,col = crop_img.shape  #crop_image 의 rwo와 col을 추출한다. 134,70
    row = 134-int(row*(3/4))  #34만큼 = 134-100  #70x134 의 이미지를 3/4씩 잘라서 cropping할 목적으로 row와 col을 구함
    col = 70-int(col*(3/4))  #18만큼 = 70-52

##원본 image 의 70*134 size로 cropping을 하는데 3/4만큼 슬라이딩을 위해서 row값을 위에서 구해주었다
#예를들어 원본 image가 134이고 이것의 3/4 는 100인데  134-100 = 34의 차이만큼 윈도우가 움직인다는 뜻이다
#그래서 34만큼 움직이는데 pyramid의 img의 row값을 34로 나누어서 최대 갈수 있는 횟수를 구하는 방식으로 구현하였다
# #col도 마찬가지이다.

    for i in range(int(img.shape[0]/row)):  #(세로 슬라이딩)
        for j in range(int(img.shape[1]/col)): #(가로 슬라이딩)

            crop_img = img[(0+row*i):(134+row*i) , (0+col*j):(70+col*j)]
            #cropping 하면서 움직일 윈도의 좌표만큼을 설정하였다.
            if crop_img.shape[0] + crop_img.shape[1] == 204:
            ##사이즈가 134 x 70 인것만 뽑아서 저장하여 70*134의 이미지만 나오도록 하였다
                prediction = sess.run(p, feed_dict={x: crop_img.reshape(1, 9380), keep_prob: 1.0})  # keep_prob : 평가
                if prediction[0,0] < prediction[0,1]: ##사진이 사람일때

                    store.append(crop_img)  #cropping 한 이미지가 배열 형식인 list에 저장이된다.

                    print("i , j = ",i,j)
                    cv2.imwrite("img"+str(num)+".png",store[num])
                    num = num + 1

############피라미드함수#################################3
def pyramid(image,multiple):  ##피라미드 함수
    row,col = image.shape  #image의 크기를 뽑아온다.
    size = col,row  #사이즈를 조절할때 col과 row의 위치를 바꿔줘야한다.
    size = int(col*multiple) , int(row*multiple)  #피라미드 영상을 곱해주고 정수변환
    image = cv2.resize(image,size)  #이미지의 크기를 resize한다
    return image #resize한 이미지를 반환한다.


#**********************************next_batch************************#
_num_examples = 1200  # 데이터 갯수
_index_in_epoch = 0  # epoch
_images = train_images  # Image 변수
_labels = train_labels  # Label 변수
_epochs_completed = 0


# batch 연산을 수행하는 함수
# 호출될 때마다 랜덤으로 batch_size의 (Image, Label) 데이터를 반환한다
def next_batch(batch_size):
    """Return the next `batch_size` examples from this data set."""
    global _index_in_epoch
    global _images
    global _labels
    global _epochs_completed

    start = _index_in_epoch
    _index_in_epoch += batch_size

    if _index_in_epoch > _num_examples:
        # Finished epoch
        _epochs_completed += 1

        # Shuffle the data
        perm = np.arange(_num_examples)
        np.random.shuffle(perm)
        _images = _images[perm]
        _labels = _labels[perm]

        # Start next epoch
        start = 0
        _index_in_epoch = batch_size
        assert batch_size <= _num_examples

    end = _index_in_epoch
    return _images[start:end], _labels[start:end]


# 가중치를 초기화하는 함수 (정규분포 stddev=0.1로 초기화한다)
def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial,name="initial")


# 바이어스를 초기화하는 함수 (0.1로 초기화한다)
def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial,name="initial2")


# 컨벌루션을 실행하는 함수
# padding = 'SAME' 입력과 출력의 이미지 크기가 같도록 해준다

# padding = 'VALID' 필터의 크기만큼 이미지 크기가 감소한다
def conv2d_valid(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='VALID')


def conv2d_same(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


# max pooling을 실행하는 함수
def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

tf.reset_default_graph()
x = tf.placeholder("float32", [None, 134*70])  # 134*70
y = tf.placeholder("float32", [None, 2])   # 클래스 개수 2개(사람 or no 사람)

W = tf.Variable(tf.zeros([134*70, 2]),name="v1")
b = tf.Variable(tf.zeros([2]),name="v2")

# 1st conv layer ----------------------
W_conv1 = weight_variable([5, 5, 1, 32])  # -4 , -4
b_conv1 = bias_variable([32])

x_image = tf.reshape(x, [-1, 134, 70, 1])

# y = x*w + b에 ReLU를 적용한다
h_conv1 = tf.nn.relu(conv2d_same(x_image, W_conv1) + b_conv1)
# (134,70) ==> (134,70)
h_pool1 = max_pool_2x2(h_conv1)
# (134,70) ==> (67, 35)

# 2nd conv layer -----------------------
W_conv2 = weight_variable([6, 6, 32, 64])  #-5 , -5
b_conv2 = bias_variable([64])

# (67, 35) ==> (62, 30)
h_conv2 = tf.nn.relu(conv2d_valid(h_pool1, W_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)
# (62, 30) ==> (31, 15)

# 1st fully connected layer -----------------------
W_fc1 = weight_variable([31 * 15 * 64, 300])
b_fc1 = bias_variable([300])

h_pool2_flat = tf.reshape(h_pool2, [-1, 31 * 15 * 64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)
# 위 연산으로 1024x1의 벡터가 생성된다


# Dropout ------------------------
keep_prob = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

# 2nd fully connected layer --------------
W_fc2 = weight_variable([300, 2])  ##라벨 클래스 개수
b_fc2 = bias_variable([2])
y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2
p=tf.nn.softmax(y_conv)

# learning_rate 잘 설정하는게 중요하다.. 0.1로 하니 전혀 변화가 없었다
learning_rate = 1e-4

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=y_conv))
optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost)

# init = tf.global_variables_initializer()
# saver = tf.train.Saver()
# sess = tf.Session()
# sess.run(init)
# saver = tf.train.Saver()

sess = tf.InteractiveSession()
sess.run(tf.global_variables_initializer())
saver = tf.train.Saver()
saver.restore(sess, "./train_session.ckpt")

# 정답률을 계산한다  y_conv  vs  y
correct_prediction = tf.equal(tf.argmax(p, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# ----------------------------------------------
batch_size = 50  # 한 루프에 몇개의 (Image, Label) 데이터를 학습하는지 설정
display_step = 20  # 루프를 돌면서 화면에 표시할 빈도 설정

for i in range(1200):
    costVal = 0.
    batch = next_batch(batch_size)
    # 20번 돌릴 때마다 결과를 확인한다
    print(i)
    if i % display_step == 0:
        train_accuracy = sess.run(accuracy, feed_dict={x: batch[0], y: batch[1], keep_prob: 1.0})
        costVal = sess.run(cost, feed_dict={x: batch[0], y: batch[1], keep_prob: 1.0})

        print('step', i, 'training_accuracy', train_accuracy, 'cost', costVal)

        # test_accuracy = sess.run(accuracy, feed_dict={x: test_images, y: test_labels, keep_prob: 1.0})
        # print('test accuracy', test_accuracy)

    # 실제 학습과정 함수, dropout 50%를 토대로 학습한다
    sess.run(optimizer, feed_dict={x: batch[0], y: batch[1], keep_prob: 0.5})

    # test_accuracy = sess.run(accuracy, feed_dict={x: test_images, y: test_labels, keep_prob: 1.0})
    # print('test accuracy', test_accuracy)

# 전부 학습이 끝나면 테스트 데이터를 넣어 정확도를 계산한다


where=saver.save(sess,"./train_session.ckpt")

test_accuracy = sess.run(accuracy, feed_dict={x: test_images, y: test_labels, keep_prob: 1.0})
print('test accuracy', test_accuracy)


############피라미드###########다양한 크기의 피라미드 영상 (85%,60%, 45%, 30%)을
#만들어서 실험할 수 있음 -결과보고서에 피라미드 영상을 몇 개를 만들어 실험하였는지 적을 것
store = [] ##항상 필수로 초기화 해줘야 한다. 왜냐하면 전의 cropping된 이미지가 저장되어 있기때문이다.
image = cv2.imread("PennPed00086.png",0)#원본 이미지를 받아온다.
cropping(image,0.5)  #이미지를 0.6배율로 cropping한다

#################가시화#####################

a = plt.subplot()
a.imshow(image,"gray")  ####원본이미지를 불러온다
#crop_img = img[(0+row*i):(134+row*i) , (0+col*j):(70+col*j)]
origin = 100/50  ######pyramid에 의해 변경된 이미지를 다시 원본으로 맞추기위한 수치(cropping에 곱해줄 값)
i=1#세로
j=5#가로  #0,0
x = (0+col*j)
x = x*origin##
y = (0+row*i)
y = y *origin##
# width = (70)+70*origin ##
width = (70)*origin
# height = (134)+134*origin ##
height = (134)* origin  ##
rect = patches.Rectangle((x,y),width,height,fill=False, edgecolor="red",linewidth=5) #(x,y), width ,height ,채우기x 라인색red 굴기=5
a.add_patch(rect)
