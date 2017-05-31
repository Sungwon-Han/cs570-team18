import tensorflow as tf
import numpy as np
import sys
import os.path
import sklearn.preprocessing as prep

# Data Part
#-------------------------------------------------------------------------
path = './result_test'                   
sample_list = os.listdir(path)
n_samples = len(sample_list)                # number of datapoints

filename = path + '/' + sample_list[0]
n_inputs = np.loadtxt(filename).size        # number of input layer variables

X_test = np.zeros([n_samples, n_inputs])    # initialization of training data array
X_test_y = np.zeros([n_samples, 5])

for i in range(n_samples):
    tempname = sample_list[i]
    filename = path + '/' + tempname
    X_test[i,:] = np.log(np.loadtxt(filename)*10000 + 1)
    X_test_y[i, int(tempname.split("_")[0]) - 1] = 1


preprocessor = prep.StandardScaler().fit(X_test)
X_test = preprocessor.transform(X_test)

#-------------------------------------------------------------------------

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
  # First Convolutional Layer
  data = tf.reshape(row_data, [-1, 267, 31, 1])   # Check!!!
  W_conv1 = weight_variable([F, T, 1, FN1])
  b_conv1 = bias_variable([FN1])
  h_conv1 = tf.nn.relu(conv2d(data, W_conv1) + b_conv1)
  h_pool1 = max_pool_4x4(h_conv1)

  # Second Convolutional Layer
  W_conv2 = weight_variable([F, T, FN1, FN2])
  b_conv2 = bias_variable([FN2])
  h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
  h_pool2 = max_pool_4x4(h_conv2)

  # First Fully Connected Layer
  _, row, col, filnum = h_pool2.shape
  W_fc1 = weight_variable([int(row * col * filnum), N1])
  b_fc1 = bias_variable([N1])
  h_pool2_flat = tf.reshape(h_pool2, [-1, int(row * col * filnum)])
  h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

  # Dropout Layer
  h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

  # Second Fully Connected Layer
  W_fc2 = weight_variable([N1, N2])
  b_fc2 = bias_variable([N2])
  h_fc2 = tf.nn.relu(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)

  # Softmax Ouput Layer
  W_fc3 = weight_variable([N2, 5])
  b_fc3 = bias_variable([5])
  y_conv=tf.nn.softmax(tf.matmul(h_fc2, W_fc3) + b_fc3)
  return y_conv


def get_random_block_from_data(data, label, batch_size, autoencoder):
    start_index = np.random.randint(0, len(data) - batch_size)
    X_batch = autoencoder.reconstruct(data[start_index:(start_index + batch_size)])
    return X_batch, label[start_index:(start_index + batch_size)]

def get_random_block_only_cnn(data, label, batch_size):
    start_index = np.random.randint(0, len(data) - batch_size)
    X_batch = data[start_index:(start_index + batch_size)]
    return X_batch, label[start_index:(start_index + batch_size)]

alpha = 1e-4
training_epochs = 120
batch_size = 64

X = tf.placeholder(tf.float32, [None, 8277])
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
saver = tf.train.Saver()
sess = tf.InteractiveSession()
sess.run(tf.global_variables_initializer())

saver.restore(sess, "./CNN_model.ckpt")
print("CNN restored.")


Test_accuracy = sess.run(accuracy, feed_dict = {X:X_test, Y_:X_test_y, keep_prob:1.0})
print("Result accuracy for test set is %g" % Test_accuracy)
