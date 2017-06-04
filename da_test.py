import tensorflow as tf
import numpy as np
import sys
import os.path
import sklearn.preprocessing as prep
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import array_ops
from cleverhans.attacks_tf import fgm

# Data Part
#-------------------------------------------------------------------------
path = './result_new'                   # directory of the folder with all of the datapoints; the folder should be in the same directory as this script
sample_list = os.listdir(path)
n_samples2 = len(sample_list)                # number of datapoints

data_indices = [[], [], [], [], []]
train_indices = []
valid_indices = []
test_indices = []

for i in range(n_samples2):
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

Train = np.concatenate((np.array(X_train), np.array(X_train_y)), axis = 1)
#temp = Train
#for i in range(20):
#    Train = np.concatenate((Train, temp), axis=0)

np.random.shuffle(Train)

X_train = Train[:, 0:8277]
X_train_y = Train[:, 8277:]

X_train = X_train.tolist()
X_train_y = X_train_y.tolist()


print("Data load finished")
#-------------------------------------------------------------------------



# Test Data part
#-------------------------------------------------------------------------
path = './result'
sample_list = os.listdir(path)
n_samples = len(sample_list)                # number of datapoints

filename = path + '/' + sample_list[0]
n_inputs = np.loadtxt(filename).size        # number of input layer variables

X_test = np.zeros([n_samples, n_inputs])    # initialization of training data array
X_test_y = np.zeros([n_samples, 5])

for i in range(n_samples):
    tempname = sample_list[i]
    filename = path + '/' + tempname
    X_test[i,:] = np.loadtxt(filename)
    X_test_y[i, int(tempname.split("_")[0]) - 1] = 1

Ad_path = './result_new'
#Ad_f = open("test_index.txt", 'r')
#Ad_sample_list = Ad_f.readlines()
#Ad_sample_list = [ _.rstrip() for _ in Ad_sample_list ]

# Adv test data
Ad_X_test = np.zeros([n_samples, n_inputs])    # initialization of training data array
Ad_X_test_y = np.zeros([n_samples, 5])

for i in range(n_samples):
    tempname = sample_list[i]
    filename = Ad_path + '/' + tempname
    Ad_X_test[i,:] = np.loadtxt(filename)
    Ad_X_test_y[i, int(tempname.split("_")[0]) - 1] = 1    

# Parameters
#-------------------------------------------------------------------------
Scale = 0.00
batch_size = 128
alpha = 4e-4
training_epochs = 140

# Network Parameters
n_hidden_1 = 1000 # 1st layer num features
n_hidden_2 = 500 # 2nd layer num features
n_input = 8277 # (img shape: 267*31)

#-------------------------------------------------------------------------

weights = {
    'encoder_h1': tf.Variable(tf.random_normal([n_input, n_hidden_1])),
    'encoder_h2': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2])),
    'decoder_h1': tf.Variable(tf.random_normal([n_hidden_2, n_hidden_1])),
    'decoder_h2': tf.Variable(tf.random_normal([n_hidden_1, n_input])),
}
biases = {
    'encoder_b1': tf.Variable(tf.random_normal([n_hidden_1])),
    'encoder_b2': tf.Variable(tf.random_normal([n_hidden_2])),
    'decoder_b1': tf.Variable(tf.random_normal([n_hidden_1])),
    'decoder_b2': tf.Variable(tf.random_normal([n_input])),
}

# Building the encoder
def encoder(x):
    # Encoder Hidden layer with sigmoid activation #1
    layer_1 = tf.nn.softplus(tf.add(tf.matmul(x, weights['encoder_h1']),
                                   biases['encoder_b1']))
    # Decoder Hidden layer with sigmoid activation #2
    layer_2 = tf.nn.softplus(tf.add(tf.matmul(layer_1, weights['encoder_h2']),
                                   biases['encoder_b2']))
    return layer_2


# Building the decoder
def decoder(x):
    # Encoder Hidden layer with sigmoid activation #1
    layer_1 = tf.nn.softplus(tf.add(tf.matmul(x, weights['decoder_h1']),
                                   		 biases['decoder_b1']))
    # Decoder Hidden layer with sigmoid activation #2
    layer_2 = tf.add(tf.matmul(layer_1, weights['decoder_h2']),
                               		biases['decoder_b2'])
    return layer_2

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

def get_random_block_only_cnn(data, label, batch_size):
    start_index = np.random.randint(0, len(data) - batch_size)
    X_batch = data[start_index:(start_index + batch_size)]
    return X_batch, label[start_index:(start_index + batch_size)]

def cnn(row_data, keep_prob, F, T, FN1, FN2, N1, N2):
  global CNN_variable1, CNN_variable2
  CNN_variable1 = {
      'W_conv1': tf.Variable(tf.truncated_normal([F, T, 1, FN1], stddev=0.1), trainable=False),
      'b_conv1': tf.Variable(tf.constant(0.1, shape=[FN1]), trainable = False),
      'W_conv2': tf.Variable(tf.truncated_normal([F, T, FN1, FN2], stddev=0.1), trainable=False),
      'b_conv2': tf.Variable(tf.constant(0.1, shape=[FN2]), trainable=False),
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
      'W_fc1': tf.Variable(tf.truncated_normal([int(row * col * filnum), N1], stddev=0.1), trainable=False),
      'b_fc1': tf.Variable(tf.constant(0.1, shape=[N1]), trainable=False),
      'W_fc2': tf.Variable(tf.truncated_normal([N1, N2], stddev=0.1), trainable=False),
      'b_fc2': tf.Variable(tf.constant(0.1, shape=[N2]), trainable=False),
      'W_fc3': tf.Variable(tf.truncated_normal([N2, 5], stddev=0.1), trainable=False),
      'b_fc3': tf.Variable(tf.constant(0.1, shape=[5]), trainable=False),
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


# tf Graph input (only pictures)
X = tf.placeholder(tf.float32, [None, 8277])
Y_ = tf.placeholder(tf.float32, [None, 5])
size = tf.placeholder(tf.int32)
keep_prob = tf.placeholder(tf.float32)

# Construct model
noise = Scale * tf.random_normal(shape = tf.shape(X))

encoder_op = encoder(X+noise)
decoder_op = decoder(encoder_op)

mean, var = tf.nn.moments(decoder_op, axes = [0], keep_dims = True)
std = tf.sqrt(var)
def _safe_div(numerator, denominator, name="value"):
  return array_ops.where(
      math_ops.greater(denominator, 0),
      math_ops.div(numerator, array_ops.where(
          math_ops.equal(denominator, 0),
          array_ops.ones_like(denominator), denominator)),
      array_ops.zeros_like(numerator),
      name=name)

scal_x = _safe_div((decoder_op - mean), tf.tile(std, [size, 1]))
Y_conv = cnn(scal_x, keep_prob, 2, 2, 10, 20, 50, 25)

# Adversarial input Test data
mean, var = tf.nn.moments(X, axes = [0], keep_dims = True)
std = tf.sqrt(var)
X_stdscale = _safe_div((X - mean), tf.tile(std, [size, 1]))
Y_conv_for_Ad = cnn(X_stdscale, keep_prob, 2, 2, 10, 20, 50, 25)

adv_x = fgm(X, Y_conv_for_Ad, y=None, eps=0.01, ord=np.inf, clip_min=None, clip_max=None)
noise = Scale * tf.random_normal(shape = tf.shape(X))

encoder_adv = encoder(adv_x + noise)
decoder_adv = decoder(encoder_adv)

mean_adv, var_adv = tf.nn.moments(decoder_adv, axes = [0], keep_dims = True)
std_adv = tf.sqrt(var_adv)

scal_x_adv = _safe_div((decoder_adv - mean_adv), tf.tile(std_adv, [size, 1]))
Y_adv = cnn(scal_x_adv, keep_prob, 2, 2, 10, 20, 50, 25)

# Train
cross_entropy = -tf.reduce_sum(Y_*tf.log(Y_conv))
train_step = tf.train.AdamOptimizer(alpha).minimize(cross_entropy)

# Accuracy
correct_prediction = tf.equal(tf.argmax(Y_conv, 1), tf.argmax(Y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# Adv Accuracy
Ad_correct_prediction = tf.equal(tf.argmax(Y_adv, 1), tf.argmax(Y_, 1))
Ad_accuracy = tf.reduce_mean(tf.cast(Ad_correct_prediction, tf.float32))

Ad_conf_mat = tf.contrib.metrics.confusion_matrix( tf.argmax(Y_, axis=1), tf.argmax(Y_adv, axis=1) )



# Initializing the variables
init = tf.global_variables_initializer()

# Launch the graph
sess = tf.Session()
loader1 = tf.train.Saver(var_list=weights)
loader2 = tf.train.Saver(var_list=biases)
loader3 = tf.train.Saver(var_list=CNN_variable1)
loader4 = tf.train.Saver(var_list=CNN_variable2)
sess.run(init)
loader1.restore(sess, "./Denoising_weights_more.ckpt")
loader2.restore(sess, "./Denoising_biases_more.ckpt")
loader3.restore(sess, "./CNN_model_variable1.ckpt")
loader4.restore(sess, "./CNN_model_variable2.ckpt")


for epoch in range(training_epochs):
    total_batch = int(n_samples2 / batch_size)
    for i in range(total_batch):
	#batch_xs, batch_ys = get_random_block_from_data(X_train, X_train_y, batch_size, autoencoder)
	batch_xs, batch_ys = get_random_block_only_cnn(X_train, X_train_y, batch_size)
	sess.run(train_step, feed_dict={X:batch_xs, Y_:batch_ys, size:batch_size, keep_prob:0.5})
    if epoch % 1 == 0:
	#valid_accuracy = accuracy.eval(feed_dict = {X:autoencoder.reconstruct(X_train), Y_:X_train_y, keep_prob:1.0})
	valid_accuracy = sess.run(accuracy, feed_dict = {X:X_validation, Y_:X_valid_y, size:len(X_validation), keep_prob:1.0})
	print("step %d, validation accuracy %g" %(epoch, valid_accuracy))

        c= sess.run(accuracy, feed_dict={X: X_test, Y_: X_test_y, size:len(X_test), keep_prob:1.0})
        Ad_c= sess.run(Ad_accuracy, feed_dict={X: Ad_X_test, Y_: Ad_X_test_y, size:len(Ad_X_test), keep_prob:1.0})

    	print("Test accuracy=", "{:.9f}".format(c))
	print("Adv Test accuracy=", "{:.9f}".format(Ad_c))
print(sess.run(Ad_conf_mat, feed_dict={X: Ad_X_test, Y_: Ad_X_test_y, size:len(Ad_X_test), keep_prob:1.0} ))




