import tensorflow.compat.v1 as tf
import numpy as np
import pandas as pd
import os, re, csv
import matplotlib.pyplot as plt

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'   # ignore Tensorflow error messages.
tf.compat.v1.disable_eager_execution()


def opencsv(filename):
    f = open(filename, 'r')
    reader = csv.reader(f)
    output = list()
    for i in reader:
        output.append(i)
    return np.array(output)


def str_to_float(field, listname):
    for i in field:
        if re.search('[a-z]', i):
            continue
        else:
            listname.append(float(re.sub(',', '', i)))


def add_mean_value(data_list_name, lag):
    target_list_name = []
    for i in range(len(data_list_name) - lag):  # 0 2897   : 2897
        # mean value of 10 data
        mean_value = 0.0
        for k in range(i, i + lag):
            mean_value += data_list_name[k]
        target_list_name.append(mean_value / lag)
    return target_list_name


# load and process
fileopen = opencsv('feeds.csv')
temperature = list()
Z = pd.DataFrame(fileopen)[2]  # temperature
str_to_float(Z, temperature)
Data = np.array(temperature)
plt.plot(Data)
plt.show()
plt.close()


y_data = []
lag = 100

x_data = add_mean_value(temperature, lag)
# for i in range(len(temperature)-lag):   # 0 2897   : 2897
#     # mean value of 10 data
#     mean_value = 0.0
#     for k in range(i, i+lag):
#         mean_value += temperature[k]
#     x_data.append(mean_value/lag)

for j in range(lag, len(temperature)):   # original: [2908] # 10 2907  : 2897
    y_data.append(temperature[j])

# Make random variables
W = tf.Variable(tf.random_uniform([1], -1.0, 1.0))  # 1 rank, vector. Random values between -1.0 ~ 1.0
b = tf.Variable(tf.random_uniform([1], -1.0, 1.0))  # 1 rank, vector. Random values between -1.0 ~ 1.0

# Define placeholder
X = tf.placeholder(tf.float32, name="X")
Y = tf.placeholder(tf.float32, name="Y")

# Hypothesis
hypothesis = W * X + b

# Loss Function ?  - Calculating loss value when the data (x, y) is given
# The lower the value, the better the performance
# Cost ?  - When calculated the value with the whole data
cost = tf.reduce_mean(tf.square(hypothesis - Y))  # MSE(Mean Squared Error)

# Optimizer
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)  # learning rate ? - How 'FAST' do you want to learn
                                                                    # Can't find best loss value if too big
                                                                    # Too slow if too small
                                                                    # These are called, 'Hyperparameter'
train_op = optimizer.minimize(cost)

# ====== We've made Linear Regression model ====== #
sess = tf.compat.v1.Session()                # Make Session object so that it can implement run()
sess.run(tf.global_variables_initializer())  # Implementing Graph should be in the session

# Implement train_op graph
for step in range(100):
    _, cost_val = sess.run([train_op, cost], feed_dict={X:x_data, Y:y_data})
    print(step, cost_val, sess.run(W), sess.run(b))

print("\n ===== predict =====\n")

# TODO: need to implement the mean value of predicted data
for times in range(len(x_data), len(temperature)):   # size of the lag
    value = sess.run(hypothesis, feed_dict={X: temperature[times]})[0]
    temperature.append(value)
    print("X: {:.2f}, Y: {:.2f}".format(temperature[times], value))

Data2 = np.array(temperature)
plt.plot(Data2)
plt.show()
plt.close()

sess.close()