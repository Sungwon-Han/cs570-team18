import tensorflow as tf
import numpy as np
import sys
import os.path
from math import sqrt
import sklearn.preprocessing as prep
from scipy.stats import bernoulli

# Parameters
alpha = 0.001
training_epochs = 30
batch_size = 128
display_step = 1
Scale = 0.01
zero_p = 0.5

# Network Parameters
n_hidden_1 = 1000 # 1st layer num features
n_hidden_2 = 500 # 2nd layer num features
n_input = 8277 # (img shape: 267*31)


# Data import
path = './result_new'     
sample_list = os.listdir(path)
n_samples = len(sample_list)   # number of datapoints

data_indices = [[], [], [], [], []]
train_indices = []
valid_indices = []

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

filename = path + '/' + sample_list[0]
n_inputs = np.loadtxt(filename).size        # number of input layer variables

X_train = np.zeros([len(train_indices), n_inputs])                 # initialization of training data array
X_validation = np.zeros([len(valid_indices), n_inputs])       # initialization of validation data array



for i in range(len(train_indices)):
    filename = path + '/' + sample_list[train_indices[i]]
    X_train[i,:] = np.loadtxt(filename)

for i in range(len(valid_indices)):
    filename = path + '/' + sample_list[valid_indices[i]]
    X_validation[i,:] = np.loadtxt(filename)

def standard_scale(X_train, X_test):
    preprocessor = prep.StandardScaler().fit(X_train)
    X_train = preprocessor.transform(X_train)
    X_test = preprocessor.transform(X_test)
    return X_train, X_test

#X_train, X_validation = standard_scale(X_train, X_validation)
#temp = X_train
#for i in range(50):
#    X_train = np.concatenate((X_train, temp), axis=0)

X_train = np.concatenate( [X_train] * 50, axis=0 )
np.random.shuffle(X_train)

def get_random_block_from_data(data, batch_size):
    start_index = np.random.randint(0, len(data) - batch_size)
    return data[start_index:(start_index + batch_size)]



weights = {
  'encoder_h1': tf.Variable(tf.random_uniform([n_input, n_hidden_1], minval= -4 * sqrt(6 / ( n_hidden_2 + n_hidden_1  ) ), maxval = 4 * sqrt(6 / (n_hidden_2 + n_hidden_1))) ),
   'encoder_h2': tf.Variable(tf.random_uniform([n_hidden_1, n_hidden_2], minval= -4 * sqrt(6 / ( n_hidden_2 + n_hidden_2  ) ), maxval = 4 * sqrt(6 / ( n_hidden_2 + n_hidden_2))) ),
   'decoder_h1': tf.Variable(tf.random_uniform([n_hidden_2, n_hidden_1], minval= -4 * sqrt(6 / ( n_hidden_2 + n_hidden_1  ) ), maxval = 4 * sqrt(6 / ( n_hidden_2 + n_hidden_1))) ),
   'decoder_h2': tf.Variable(tf.random_uniform([n_hidden_1, n_input] , minval= -4 * sqrt(6 / ( n_input + n_hidden_1  ) ), maxval = 4 * sqrt(6 / ( n_input + n_hidden_1) ) )),
}

biases = {
    'encoder_b1': tf.Variable(tf.random_uniform([n_hidden_1])),
    'encoder_b2': tf.Variable(tf.random_uniform([n_hidden_2])),
    'decoder_b1': tf.Variable(tf.random_uniform([n_hidden_1])),
    'decoder_b2': tf.Variable(tf.random_uniform([n_input])),
}


# Building the encoder
def encoder(x):
    # Encoder Hidden layer with sigmoid activation #1
    layer_1 = tf.nn.softplus(tf.add(tf.matmul(x, weights['encoder_h1']),
                                   		 biases['encoder_b1']))

    # Decoder Hidden layer with sigmoid activation #2
    layer_2 = tf.nn.softplus(tf.add(tf.matmul(layer_1, weights['encoder_h2']),
                                   		       biases['encoder_b2']))
        # Encoder Hidden layer with sigmoid activation #1
    #layer_1 = tf.nn.softplus(tf.matmul(x, weights['encoder_h1']))
#    # Decoder Hidden layer with sigmoid activation #2
    #layer_2 = tf.nn.softplus(tf.matmul(layer_1, weights['encoder_h2']))

    return layer_2


# Building the decoder
def decoder(x):
    # Encoder Hidden layer with sigmoid activation #1
    layer_1 = tf.add(tf.matmul(x, weights['decoder_h1']),
                                  biases['decoder_b1'])

   # layer_1_drop = tf.nn.dropout(layer_1, 0.5)
   # Decoder Hidden layer with sigmoid activation #2
   # layer_2 = tf.add(tf.matmul(layer_1, weights['decoder_h2']),
   #                            		     biases['decoder_b2'])
    # Encoder Hidden layer with sigmoid activation #1
   # layer_1 = tf.nn.softplus(tf.matmul(x, weights['decoder_h1']))
#    # Decoder Hidden layer with sigmoid activation #2
#    layer_2 = tf.matmul(layer_1, weights['decoder_h2'])
    layer_2 = tf.nn.sigmoid(tf.add(tf.matmul(layer_1, weights['decoder_h2']),
                           			      biases['decoder_b2']))
    return layer_2

# Construct model
X = tf.placeholder("float", [None, n_input])
zerolist = tf.placeholder("float", [None, n_input])
X_removed = X * zerolist
X_concat = tf.concat((X, X_removed), axis=0)
noise = Scale * tf.random_normal(shape = tf.shape(X_concat))
X_noise = X_concat + noise
X_clip = tf.clip_by_value(X_noise, 0, 1)

encoder_op = encoder(X_clip)
decoder_op = decoder(encoder_op)

# Prediction
y_pred = decoder_op
# Targets (Labels) are the input data.
y_true = tf.concat((X, X), axis=0)

# Define loss and optimizer, minimize the squared error
cost = tf.reduce_mean(tf.pow(y_true - y_pred, 2))
#cost = -tf.reduce_sum(y_true*tf.log(y_pred))
optimizer = tf.train.AdamOptimizer(learning_rate = alpha).minimize(cost)

# Initializing the variables
init = tf.global_variables_initializer()

# Launch the graph
sess = tf.Session()
saver1 = tf.train.Saver(var_list = weights)
saver2 = tf.train.Saver(var_list = biases)
sess.run(init)

total_batch = int(n_samples / batch_size * 50)
zero_list = bernoulli.rvs(1-zero_p, size = n_input * len(X_validation))
zero_list = np.reshape(zero_list, [-1, n_input])
valcost = sess.run(cost, feed_dict={X: X_validation, zerolist:zero_list})
print("Epoch:", '0000', "cost=", "{:.9f}".format(valcost))
# Training cycle
print(X_validation)
for epoch in range(training_epochs):
    # Loop over all batches
    for i in range(total_batch):
	zero_list = bernoulli.rvs(1-zero_p, size = n_input * batch_size)
	zero_list = np.reshape(zero_list, [-1, n_input])
        batch_xs = get_random_block_from_data(X_train, batch_size)
        _, traincost = sess.run([optimizer, cost], feed_dict={X: batch_xs, zerolist: zero_list})

    # Display logs per epoch step
    if epoch % display_step == 0:
	zero_list = bernoulli.rvs(1-zero_p, size = n_input * len(X_validation))
	zero_list = np.reshape(zero_list, [-1, n_input])
	valcost = sess.run(cost, feed_dict={X: X_validation, zerolist:zero_list})
        print("Epoch:", '%04d' % (epoch+1),
                  "valcost=", "{:.9f}".format(valcost), "traincost=", "{:.9f}".format(traincost))

    	print(sess.run(decoder_op, feed_dict={X: X_validation, zerolist:zero_list}))

save_path = saver1.save(sess, "Denoising_weights_more.ckpt")
save_path2 = saver2.save(sess, "Denoising_biases_more.ckpt")

