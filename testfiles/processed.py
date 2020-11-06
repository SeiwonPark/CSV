import pandas as pd
import csv
import re
import matplotlib.pyplot as plt


def opencsv(filename):
    f = open(filename, 'r')
    reader = csv.reader(f)
    output = []
    for i in reader:
        output.append(i)
    return output


def str_to_float(field, listname):
    for i in field:
        if re.search('[a-z]', i):
            continue
        else:
            listname.append(float(re.sub(',', '', i)))


fileopen = opencsv('feeds.csv')

field1 = pd.DataFrame(fileopen)[2]  # temperature
field2 = pd.DataFrame(fileopen)[3]  # humidity
field3 = pd.DataFrame(fileopen)[4]  # pressure
field4 = pd.DataFrame(fileopen)[5]  # PM 1.0
field5 = pd.DataFrame(fileopen)[6]  # PM 2.5
field6 = pd.DataFrame(fileopen)[7]  # PM 10

temperature, humidity, pressure, pm1_0, pm2_5, pm10 = [], [], [], [], [], []

str_to_float(field1, temperature)
str_to_float(field2, humidity)
str_to_float(field3, pressure)
str_to_float(field4, pm1_0)
str_to_float(field5, pm2_5)
str_to_float(field6, pm10)

# entry = list(range(1, len(temperature) + 1))
# plt.scatter(entry, temperature)
# plt.scatter(entry, humidity)
# plt.xlabel('Entry')
# plt.ylabel('Value')
# plt.show()
# plt.close()
# plt.plot(entry, pressure)
# plt.xlabel('Entry')
# plt.ylabel('Value')
# plt.show()
# plt.close()
# plt.plot(entry, pm1_0)
# plt.plot(entry, pm2_5)
# plt.plot(entry, pm10)
# plt.xlabel('Entry')
# plt.ylabel('Value')
# plt.show()

x = list(range(1, len(temperature) + 1))
y = temperature[:]

# w = 1.0
# b = 1.0
#
# y_hat = x[0] * w + b
#
# w_inc = w + 0.1
# y_hat_inc = x[0] * w_inc + b
#
# w_rate = (y_hat_inc - y_hat) / (w_inc - w)
#
# w_new = w + w_rate
#
# b_inc = b + 0.1
# y_hat_inc = x[0] * w + b_inc
#
# b_rate = (y_hat_inc - y_hat) / (b_inc - b)
#
# b_new = b + 1
#
# err = y[0] - y_hat
# w_new = w + w_rate * err
# b_new = b + 1 * err
#
# y_hat = x[1] * w_new + b_new
# err = y[1] - y_hat
# w_rate = x[1]
# w_new += w_rate * err
# b_new += 1 * err
#
# for x_i, y_i in zip(x, y):
#     y_hat = x_i * w + b
#     err = y_i - y_hat
#     w_rate = x_i
#     w += w_rate * err
#     b += 1 * err
#
# print(w, b)


class Neuron:

    def __init__(self):
        self.w = 1.0
        self.b = 1.0

    def forpass(self, x):
        y_hat = x * self.w + self.b
        return y_hat

    def backprop(self, x, err):
        w_grad = x * err
        b_grad = 1 * err
        return w_grad, b_grad

    def fit(self, x, y, epochs=100):
        for i in range(epochs):
            for x_i, y_i in zip(x, y):
                y_hat = self.forpass(x_i)
                err = -(y_i - y_hat)
                w_grad, b_grad = self.backprop(x_i, err)
                self.w -= w_grad
                self.b -= b_grad


neuron = Neuron()
neuron.fit(x, y)

plt.plot(x,y)
pt1 = (-0.1, -0.1 * neuron.w + neuron.b)
pt2 = (0.15, 0.15 * neuron.w + neuron.b)
plt.plot([pt1[0], pt2[0]], [pt1[1], pt2[1]])
plt.xlabel('x')
plt.ylabel('y')
plt.show()

