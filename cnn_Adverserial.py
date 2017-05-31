from cleverhans.attacks_tf import fgm
import tensorflow as tf
import numpy as np
import sys
import os.path
import sklearn.preprocessing as prep
from DenoisingAutoencoder import AdditiveGaussianNoiseAutoencoder

# Data Part
#-------------------------------------------------------------------------
path = './result_win3000'                   # directory of the folder with all of the datapoints; the folder should be in the same directory as this script
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
    X_train[i,:] = np.log(np.loadtxt(filename)*10000 + 1)
    X_train_y[i, int(tempname.split("_")[0]) - 1] = 1

for i in range(len(valid_indices)):
    tempname = sample_list[valid_indices[i]]
    filename = path + '/' + tempname
    X_validation[i,:] = np.log(np.loadtxt(filename)*10000 + 1)
    X_valid_y[i, int(tempname.split("_")[0]) - 1] = 1

def standard_scale(X_train, X_test):
    preprocessor = prep.StandardScaler().fit(X_train)
    X_train = preprocessor.transform(X_train)
    X_test = preprocessor.transform(X_test)
    return X_train, X_test

Train = np.concatenate((np.array(X_train), np.array(X_train_y)), axis = 1)
np.random.shuffle(Train)

X_train = Train[:, 0:10980]
X_train_y = Train[:, 10980:]

X_train = X_train.tolist()
X_train_y = X_train_y.tolist()

X_train, X_validation = standard_scale(X_train, X_validation)

print("Data load finished")
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
  data = tf.reshape(row_data, [-1, 366, 30, 1])   # Check!!!
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

# Autoencoder & CNN parameters
#-------------------------------------------------------------------------
n_inputs = 10980
hidden_layers = [200, 100]
Scale = 0.7
alpha = 1e-4
training_epochs = 120
batch_size = 64

Adv_eps = 0.01
#-------------------------------------------------------------------------

# Model Part
#autoencoder = AdditiveGaussianNoiseAutoencoder(n_input = n_inputs,
#					       n_hiddens = hidden_layers,
#					       transfer_function = tf.nn.softplus,
#					       optimizer = tf.train.AdamOptimizer(learning_rate = 0.001),
#					       scale = Scale)

X = tf.placeholder(tf.float32, [None, 10980])
Y_ = tf.placeholder(tf.float32, [None, 5])
keep_prob = tf.placeholder(tf.float32)
Y_conv = cnn(X, keep_prob, 2, 2, 10, 20, 50, 25)

adv_x = fgm(X, Y_conv, y=None, eps=Adv_eps, ord=np.inf, clip_min=None, clip_max=None)
Ad_Y_conv = cnn(adv_x, keep_prob, 2, 2, 10, 20, 50, 25)

# Train
cross_entropy = -tf.reduce_sum(Y_*tf.log(Y_conv))
train_step = tf.train.AdamOptimizer(alpha).minimize(cross_entropy)

# prediction
correct_prediction = tf.equal(tf.argmax(Y_conv, 1), tf.argmax(Y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# prediction of Adv input
Ad_correct_prediction = tf.equal(tf.argmax(Ad_Y_conv, 1), tf.argmax(Y_, 1))
Ad_accuracy = tf.reduce_mean(tf.cast(Ad_correct_prediction, tf.float32))

"""
Train & Save Model
"""
#loader = tf.train.Saver(var_list = autoencoder.weights)
saver = tf.train.Saver()
sess = tf.InteractiveSession()
sess.run(tf.global_variables_initializer())

#loader.restore(sess, "trained/autoencoder.ckpt") # Check!!
#print("Autoencoder restored.")

for epoch in range(training_epochs):
    avg_cost = 0.
    total_batch = int(n_samples / batch_size)
    for i in range(total_batch):
    	#batch_xs, batch_ys = get_random_block_from_data(X_train, X_train_y, batch_size, autoencoder)
    	batch_xs, batch_ys = get_random_block_only_cnn(X_train, X_train_y, batch_size)
    	sess.run(train_step, feed_dict={X:batch_xs, Y_:batch_ys, keep_prob:0.5})

    if epoch % 20 == 0:
    	#valid_accuracy = accuracy.eval(feed_dict = {X:autoencoder.reconstruct(X_train), Y_:X_train_y, keep_prob:1.0})
    	valid_accuracy = sess.run(accuracy, feed_dict = {X:X_validation, Y_:X_valid_y, keep_prob:1.0})
    	print("step %d, training accuracy %g" %(epoch, valid_accuracy))


print("validation accuracy %g" % sess.run(accuracy, feed_dict={X:X_validation, Y_:X_valid_y, keep_prob: 1.0}))
print("Adv validation accuracy %g" % sess.run(Ad_accuracy, feed_dict={X:X_validation, Y_:X_valid_y, keep_prob: 1.0}))


#save_path = saver.save(sess, "CNN_model.ckpt")
