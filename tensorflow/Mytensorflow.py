import numpy
import h5py
import matplotlib.pyplot as plt
import tensorflow as tf
import gzip

import numpy
from six.moves import xrange  # pylint: disable=redefined-builtin

from tensorflow.contrib.learn.python.learn.datasets import base
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import random_seed
import zipfile

# with h5py.File("kalph_train.hf", "r") as hf:
#     images = numpy.array(hf["images"])
#     labels = numpy.array(hf["labels"])
# num_imgs, rows, cols = images.shape
#
# with h5py.File("kalph_test.hf", "r") as hf:
#     images1 = numpy.array(hf["images"])
#     labels1 = numpy.array(hf["labels"])
# num_imgs, rows, cols = images1.shape
# print(num_imgs)
# print(images.shape) #19600, 52 ,52
# print(images1.shape) #19600, 52 ,52
# print(labels) # 19600
# images = images.reshape(num_imgs, rows, cols, 1)
# labels = dense_to_one_hot(labels,14)



class DataSet(object):

  def __init__(self,
               images,
               labels,
               fake_data=False,
               one_hot=False,
               dtype=dtypes.float32,
               reshape=True,
               seed=None):
    """Construct a DataSet.
    one_hot arg is used only if fake_data is true.  `dtype` can be either
    `uint8` to leave the input as `[0, 255]`, or `float32` to rescale into
    `[0, 1]`.  Seed arg provides for convenient deterministic testing.
    """
    seed1, seed2 = random_seed.get_seed(seed)
    # If op level seed is not set, use whatever graph level seed is returned
    numpy.random.seed(seed1 if seed is None else seed2)
    dtype = dtypes.as_dtype(dtype).base_dtype
    if dtype not in (dtypes.uint8, dtypes.float32):
      raise TypeError('Invalid image dtype %r, expected uint8 or float32' %
                      dtype)
    if fake_data:
      self._num_examples = 10000
      self.one_hot = one_hot
    else:
      assert images.shape[0] == labels.shape[0], (
          'images.shape: %s labels.shape: %s' % (images.shape, labels.shape))
      self._num_examples = images.shape[0]

      # Convert shape from [num examples, rows, columns, depth]
      # to [num examples, rows*columns] (assuming depth == 1)
      if reshape:
        assert images.shape[3] == 1
        images = images.reshape(images.shape[0],
                                images.shape[1] * images.shape[2])
      if dtype == dtypes.float32:
        # Convert from [0, 255] -> [0.0, 1.0].
        images = images.astype(numpy.float32)
        images = numpy.multiply(images, 1.0 / 255.0)
    self._images = images
    self._labels = labels
    self._epochs_completed = 0
    self._index_in_epoch = 0

  @property
  def images(self):
    return self._images

  @property
  def labels(self):
    return self._labels

  @property
  def num_examples(self):
    return self._num_examples

  @property
  def epochs_completed(self):
    return self._epochs_completed

  def next_batch(self, batch_size, fake_data=False, shuffle=True):
    """Return the next `batch_size` examples from this data set."""
    if fake_data:
      fake_image = [1] * 784
      if self.one_hot:
        fake_label = [1] + [0] * 14
      else:
        fake_label = 0
      return [fake_image for _ in xrange(batch_size)], [
          fake_label for _ in xrange(batch_size)
      ]
    start = self._index_in_epoch
    # Shuffle for the first epoch
    if self._epochs_completed == 0 and start == 0 and shuffle:
      perm0 = numpy.arange(self._num_examples)
      numpy.random.shuffle(perm0)
      self._images = self.images[perm0]
      self._labels = self.labels[perm0]
    # Go to the next epoch
    if start + batch_size > self._num_examples:
      # Finished epoch
      self._epochs_completed += 1
      # Get the rest examples in this epoch
      rest_num_examples = self._num_examples - start
      images_rest_part = self._images[start:self._num_examples]
      labels_rest_part = self._labels[start:self._num_examples]
      # Shuffle the data
      if shuffle:
        perm = numpy.arange(self._num_examples)
        numpy.random.shuffle(perm)
        self._images = self.images[perm]
        self._labels = self.labels[perm]
      # Start next epoch
      start = 0
      self._index_in_epoch = batch_size - rest_num_examples
      end = self._index_in_epoch
      images_new_part = self._images[start:end]
      labels_new_part = self._labels[start:end]
      return numpy.concatenate((images_rest_part, images_new_part), axis=0) , numpy.concatenate((labels_rest_part, labels_new_part), axis=0)
    else:
      self._index_in_epoch += batch_size
      end = self._index_in_epoch
      return self._images[start:end], self._labels[start:end]


def read_data_sets(fake_data=False,
                   one_hot=False,
                   dtype=dtypes.float32,
                   reshape=True,
                   validation_size=5000,
                   seed=None):


  if fake_data:

    def fake():
      return DataSet(
          [], [], fake_data=True, one_hot=one_hot, dtype=dtype, seed=seed)

    train = fake()
    validation = fake()
    test = fake()
    return base.Datasets(train=train, validation=validation, test=test)

  with h5py.File("resized_kalph_train.hf", "r") as hf:
      images = numpy.array(hf["images"])
      labels = numpy.array(hf["labels"])
  num_imgs, rows, cols = images.shape
  train_images = images.reshape(num_imgs, rows, cols, 1)
  train_labels = dense_to_one_hot(labels, 14)
  print("training = ",train_images.shape)

  with h5py.File("resized_kalph_test.hf", "r") as hf:
      images1 = numpy.array(hf["images"])
      labels1 = numpy.array(hf["labels"])
  num_imgs, rows, cols = images1.shape
  test_images = images1.reshape(num_imgs,rows,cols,1)
  test_labels = dense_to_one_hot(labels1,14)
  print("training = ", test_images.shape)

  # print(images.shape) #19600, 52 ,52
  # print(labels) # 19600
  # images = images.reshape(num_imgs, rows, cols, 1)
  # labels = dense_to_one_hot(labels,14)

  if not 0 <= validation_size <= len(train_images):
    raise ValueError(
        'Validation size should be between 0 and {}. Received: {}.'
        .format(len(train_images), validation_size))

  validation_images = train_images[:validation_size]
  validation_labels = train_labels[:validation_size]
  train_images = train_images[validation_size:]
  train_labels = train_labels[validation_size:]

  train = DataSet(
      train_images, train_labels, dtype=dtype, reshape=reshape, seed=seed)
  validation = DataSet(
      validation_images,
      validation_labels,
      dtype=dtype,
      reshape=reshape,
      seed=seed)
  test = DataSet(
      test_images, test_labels, dtype=dtype, reshape=reshape, seed=seed)

  return base.Datasets(train=train, validation=validation, test=test)


def next_batch(self, batch_size, fake_data=False, shuffle=True):
    """Return the next `batch_size` examples from this data set."""
    if fake_data:
        fake_image = [1] * 784
        if self.one_hot:
            fake_label = [1] + [0] * 13
        else:
            fake_label = 0
        return [fake_image for _ in xrange(batch_size)], [
            fake_label for _ in xrange(batch_size)
        ]
    start = self._index_in_epoch
    # Shuffle for the first epoch
    if self._epochs_completed == 0 and start == 0 and shuffle:
        perm0 = numpy.arange(self._num_examples)
        numpy.random.shuffle(perm0)
        self._images = self.images[perm0]
        self._labels = self.labels[perm0]
    # Go to the next epoch
    if start + batch_size > self._num_examples:
        # Finished epoch
        self._epochs_completed += 1
        # Get the rest examples in this epoch
        rest_num_examples = self._num_examples - start
        images_rest_part = self._images[start:self._num_examples]
        labels_rest_part = self._labels[start:self._num_examples]
        # Shuffle the data
        if shuffle:
            perm = numpy.arange(self._num_examples)
            numpy.random.shuffle(perm)
            self._images = self.images[perm]
            self._labels = self.labels[perm]
        # Start next epoch
        start = 0
        self._index_in_epoch = batch_size - rest_num_examples
        end = self._index_in_epoch
        images_new_part = self._images[start:end]
        labels_new_part = self._labels[start:end]
        return numpy.concatenate((images_rest_part, images_new_part), axis=0), numpy.concatenate(
            (labels_rest_part, labels_new_part), axis=0)
    else:
        self._index_in_epoch += batch_size
        end = self._index_in_epoch
        return self._images[start:end], self._labels[start:end]


def dense_to_one_hot(labels_dense, num_classes):
  """Convert class labels from scalars to one-hot vectors."""
  num_labels = labels_dense.shape[0]
  index_offset = numpy.arange(num_labels) * num_classes
  labels_one_hot = numpy.zeros((num_labels, num_classes))
  labels_one_hot.flat[index_offset + labels_dense.ravel()] = 1
  return labels_one_hot



# print("num_imgs = ",num_imgs)
# print("rows = ",rows)
# print("cols = ",cols)
# print(labels[0])
# print((images))
# print(labels)
# plt.imshow(images[1], cmap="gray_r")
# plt.axis("off")
# plt.show()
han =read_data_sets(one_hot=True)

with h5py.File("kalph_test.hf", "r") as hf:
    images1 = numpy.array(hf["images"])
    labels1 = numpy.array(hf["labels"])
num_imgs, rows, cols = images1.shape
test_images = images1.reshape(num_imgs, rows, cols,1)
test_labels = dense_to_one_hot(labels1, 14)

def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)
def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)
def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')
def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],strides=[1, 2, 2, 1], padding='SAME')

x_input = tf.placeholder("float",shape=[None,784])
y_input = tf.placeholder("float",shape=[None,14])

W_conv1 = weight_variable([5, 5, 1, 32])
b_conv1 = bias_variable([32])

x_image = tf.reshape(x_input, [-1,28,28,1])

h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1)

W_conv2 = weight_variable([5, 5, 32, 64])
b_conv2 = bias_variable([64])
h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)

W_conv = weight_variable([5, 5, 64, 128])
b_conv= bias_variable([128])
h_conv3 = tf.nn.relu(conv2d(h_conv2, W_conv) + b_conv)

h_pool2 = max_pool_2x2(h_conv3)

W_fc1 = weight_variable([7*7*128, 1024])
b_fc1 = bias_variable([1024])
h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*128])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

keep_prob = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

W_fc2 = weight_variable([1024, 14])
b_fc2 = bias_variable([14])
y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2

cross_entropy = tf.reduce_mean(
tf.nn.softmax_cross_entropy_with_logits(labels=y_input, logits=y_conv))
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_input,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

sess = tf.InteractiveSession()
sess.run(tf.global_variables_initializer())
for i in range(7000):
    batch = han.train.next_batch(50)
    if i%100 == 0:
        train_accuracy = accuracy.eval(feed_dict={x_input:batch[0], y_input: batch[1], keep_prob: 1.0})
        print('step', i, 'training accuracy', train_accuracy)
    train_step.run(feed_dict={x_input: batch[0], y_input: batch[1], keep_prob: 0.5})

test_accuracy = accuracy.eval(feed_dict={x_input: han.test.images , y_input: han.test.labels, keep_prob: 1.0})

print("test accuracy", test_accuracy)

#**************************

import zipfile
from io import StringIO
import io
from PIL import Image
import imghdr
import numpy as np

imgzip = open('100-Test.zip')
zippedImgs = zipfile.ZipFile(imgzip)


with zipfile.ZipFile('train_image.zip',"r") as myzip:

    for i in range(1,len(myzip.namelist())):
        print("iter ", i)
        file_in_zip = myzip.namelist()[i]

        if (".bmp" in file_in_zip or ".JPG" in file_in_zip):
            print("Found image: ", myzip.open(file_in_zip))
            data = myzip.read(file_in_zip)
            dataEnc = io.BytesIO(data)
            img = Image.open(dataEnc)

        else:
            print("???")

with zipfile.ZipFile('train_image.zip', "r") as myzip:
    for i in range(1, len(myzip.namelist())):
        print("iter ", i)
        file_in_zip = myzip.namelist()[i]


#****************************
import glob
import tensorflow as tf
import cv2
import numpy as np
train_data = np.array([]) #빈 배열 생성
test_data = np.array([])

for i in range(1, 701):
    bmp = "train_image/train_" + str(i) + ".bmp"
    img = cv2.imread(bmp)
    train_data = np.append(train_data,np.asarray(img,dtype='float'))
train_data = train_data.reshape(700,55,40,3)

for i in range(1, 701):
    bmp = "test_image/test_" + str(i) + ".bmp"
    img = cv2.imread(bmp)
    test_data = np.append(test_data,np.asarray(img,dtype='float'))
test_data = test_data.reshape(700,55,40,3)

with open('train_label.txt') as f:
    content = f.readlines()
content = [x.strip() for x in content]  ##리스트 형태로 들어간다.
x =np.array(content,dtype='float')

with open('test_label.txt') as f:
    content = f.readlines()
content = [x.strip() for x in content]
y =np.array(content,dtype='float')


train_label= np.zeros((700,100))
test_label = np.zeros((700,100))
j=0
for i in range(0,x.__len__()):
    train_label[i][j] = 1
    print(i)
    if ((i+1) % 7) == 0:
        j = j + 1

j=0
for i in range(0,x.__len__()):
    test_label[i][j] = 1
    print(i)
    if ((i+1) % 7) == 0:
        j = j + 1

def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)
def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)
def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')
def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],strides=[1, 2, 2, 1], padding='SAME')

x_input = tf.placeholder("float",shape=[None,6600])
y_input = tf.placeholder("float",shape=[None,100])

W_conv1 = weight_variable([5, 5, 3, 32])
b_conv1 = bias_variable([32])

x_image = tf.reshape(x_input, [-1,55,40,3])

h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1)

W_conv2 = weight_variable([5, 5, 32, 64])
b_conv2 = bias_variable([64])
h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)

W_fc1 = weight_variable([14*10*64, 1024])
b_fc1 = bias_variable([1024])
h_pool2_flat = tf.reshape(h_pool2, [-1, 13*13*64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

keep_prob = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

W_fc2 = weight_variable([1024, 100])
b_fc2 = bias_variable([100])
y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2
cross_entropy = tf.reduce_mean(
tf.nn.softmax_cross_entropy_with_logits(labels=y_input, logits=y_conv))
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_input,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

sess = tf.InteractiveSession()
sess.run(tf.global_variables_initializer())
for i in range(7000):
    #batch = han.train.next_batch(50)
    image_batch, label_batch = tf.train.batch([train_data, train_label],batch_size=50)
    if i%100 == 0:
        train_accuracy = accuracy.eval(feed_dict={x_input:image_batch, y_input: label_batch, keep_prob: 1.0})
        print('step', i, 'training accuracy', train_accuracy)
    train_step.run(feed_dict={x_input: image_batch, y_input: label_batch, keep_prob: 0.5})

test_accuracy = accuracy.eval(feed_dict={x_input: test_data , y_input: test_label, keep_prob: 1.0})


