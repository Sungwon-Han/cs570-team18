import tensorflow as tf
import numpy as np
import sys
import os.path
import sklearn.preprocessing as prep
from DenoisingAutoencoder import AdditiveGaussianNoiseAutoencoder

# Data Part
#-------------------------------------------------------------------------
path = './result_new'                   # directory of the folder with all of the datapoints; the folder should be in the same directory as this script
sample_list = os.listdir(path)
n_samples = len(sample_list)                # number of datapoints

data_indices = [[], [], [], [], []]
train_indices = []
valid_indices = []
test_indices = []

for i in range(n_samples):
    filename = sample_list[i]
    file_list = filename.split("_")
    data_indices[int(file_list[0])-1].append(i)

for j in data_indices:
    length = len(j)
    n_train = int(round(length * 0.6))  	 # number of training points:  	 60.0% of all datapoints
    n_valid = int(round(length * 0.2))		 # number of validation point:	 20.0% of all datapoints
    train_indices.extend(j[0: n_train])
    valid_indices.extend(j[n_train: n_train + n_valid])
    test_indices.extend(j[n_train + n_valid :])


filename = path + '/' + sample_list[0]
n_inputs = np.loadtxt(filename).size        # number of input layer variables

X_train = np.zeros([len(train_indices), n_inputs])    # initialization of training data array
X_validation = np.zeros([len(valid_indices), n_inputs])       # initialization of validation data array
X_train_y = np.zeros([len(train_indices), 5])
X_valid_y = np.zeros([len(valid_indices), 5])


for i in range(len(train_indices)):
    tempname = sample_list[train_indices[i]]
    filename = path + '/' + tempname
    X_train[i,:] = np.loadtxt(filename)
    X_train_y[i, int(tempname.split("_")[0]) - 1] = 1

for i in range(len(valid_indices)):
    tempname = sample_list[valid_indices[i]]
    filename = path + '/' + tempname
    X_validation[i,:] = np.loadtxt(filename)
    X_valid_y[i, int(tempname.split("_")[0]) - 1] = 1

def standard_scale(X_train, X_test):
    preprocessor = prep.StandardScaler().fit(X_train)
    X_train = preprocessor.transform(X_train)
    X_test = preprocessor.transform(X_test)
    return X_train, X_test

Train = np.concatenate((np.array(X_train), np.array(X_train_y)), axis = 1)
np.random.shuffle(Train)

X_train = Train[:, 0:8277]
X_train_y = Train[:, 8277:]

X_train = X_train.tolist()
X_train_y = X_train_y.tolist()

X_train, X_validation = standard_scale(X_train, X_validation)

print("Data load finished")


def weight_variable(shape):
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)

def bias_variable(shape):
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)

def conv2d(x, W):
  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


def max_pool_4x4(x):
  return tf.nn.max_pool(x, ksize=[1, 4, 4, 1],
                        strides=[1, 2, 2, 1], padding='SAME')

def cnn(row_data, keep_prob, F, T, FN1, FN2, N1, N2):
  global CNN_variable1, CNN_variable2
  CNN_variable1 = {
      'W_conv1': tf.Variable(tf.truncated_normal([F, T, 1, FN1], stddev=0.1)),
      'b_conv1': tf.Variable(tf.constant(0.1, shape=[FN1])),
      'W_conv2': tf.Variable(tf.truncated_normal([F, T, FN1, FN2], stddev=0.1)),
      'b_conv2': tf.Variable(tf.constant(0.1, shape=[FN2])),
  }
  # First Convolutional Layer
  data = tf.reshape(row_data, [-1, 267, 31, 1])   # Check!!!
  W_conv1 = CNN_variable1['W_conv1']
  b_conv1 = CNN_variable1['b_conv1']
  h_conv1 = tf.nn.relu(conv2d(data, W_conv1) + b_conv1)
  h_pool1 = max_pool_4x4(h_conv1)

  # Second Convolutional Layer
  W_conv2 = CNN_variable1['W_conv2']
  b_conv2 = CNN_variable1['b_conv2']
  h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
  h_pool2 = max_pool_4x4(h_conv2)

  # First Fully Connected Layer
  _, row, col, filnum = h_pool2.shape

  CNN_variable2 = {
      'W_fc1': tf.Variable(tf.truncated_normal([int(row * col * filnum), N1], stddev=0.1)),
      'b_fc1': tf.Variable(tf.constant(0.1, shape=[N1])),
      'W_fc2': tf.Variable(tf.truncated_normal([N1, N2], stddev=0.1)),
      'b_fc2': tf.Variable(tf.constant(0.1, shape=[N2])),
      'W_fc3': tf.Variable(tf.truncated_normal([N2, 5], stddev=0.1)),
      'b_fc3': tf.Variable(tf.constant(0.1, shape=[5])),
  }

  W_fc1 = CNN_variable2['W_fc1']
  b_fc1 = CNN_variable2['b_fc1']
  h_pool2_flat = tf.reshape(h_pool2, [-1, int(row * col * filnum)])
  h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

  # Dropout Layer
  h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

  # Second Fully Connected Layer
  W_fc2 = CNN_variable2['W_fc2']
  b_fc2 = CNN_variable2['b_fc2']
  h_fc2 = tf.nn.relu(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)

  # Softmax Ouput Layer
  W_fc3 = CNN_variable2['W_fc3']
  b_fc3 = CNN_variable2['b_fc3']
  y_conv=tf.nn.softmax(tf.matmul(h_fc2, W_fc3) + b_fc3)
  return y_conv


def get_random_block_only_cnn(data, label, batch_size):
    start_index = np.random.randint(0, len(data) - batch_size)
    X_batch = data[start_index:(start_index + batch_size)]
    return X_batch, label[start_index:(start_index + batch_size)]

n_inputs = 8277
alpha = 1e-4
training_epochs = 45
batch_size = 64

X = tf.placeholder(tf.float32, [None, n_inputs])
Y_ = tf.placeholder(tf.float32, [None, 5])
keep_prob = tf.placeholder(tf.float32)
Y_conv = cnn(X, keep_prob, 2, 2, 10, 20, 50, 25)	

cross_entropy = -tf.reduce_sum(Y_*tf.log(Y_conv))
train_step = tf.train.AdamOptimizer(alpha).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(Y_conv, 1), tf.argmax(Y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

"""
Train & Save Model
"""
saver1 = tf.train.Saver(var_list = CNN_variable1)
saver2 = tf.train.Saver(var_list = CNN_variable2)
sess = tf.InteractiveSession()
sess.run(tf.global_variables_initializer())


for epoch in range(training_epochs):
    avg_cost = 0.
    total_batch = int(n_samples / batch_size)
    for i in range(total_batch):
	batch_xs, batch_ys = get_random_block_only_cnn(X_train, X_train_y, batch_size)
	sess.run(train_step, feed_dict={X:batch_xs, Y_:batch_ys, keep_prob:0.5})

    if epoch % 20 == 0:
	valid_accuracy = sess.run(accuracy, feed_dict = {X:X_validation, Y_:X_valid_y, keep_prob:1.0})
	print("step %d, validation accuracy %g" %(epoch, valid_accuracy))

print("validation accuracy %g" % sess.run(accuracy, feed_dict={X:X_validation, Y_:X_valid_y, keep_prob: 1.0}))

save_path1 = saver1.save(sess, "CNN_model_variable1.ckpt")
save_path2 = saver2.save(sess, "CNN_model_variable2.ckpt")
