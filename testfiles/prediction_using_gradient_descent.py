import csv
import re
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

# load dataset
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


# load and process
fileopen = opencsv('feeds.csv')
temperature = list()
load = pd.DataFrame(fileopen)[4]  # temperature
str_to_float(load, temperature)
original_data = np.array(temperature)

data_lag = len(original_data)//2
x = original_data[:data_lag]
y = original_data[data_lag:data_lag * 2]

w = 0
b = 0

lr = 0.0001
epochs = 100
n = float(len(x))

# Performing Gradient Descent
for i in range(epochs):
    y_hat = w * x + b
    D_w = (-2 / n) * sum(x * (y - y_hat))  # Derivative wrt w
    D_b = (-2 / n) * sum(y - y_hat)  # Derivative wrt b
    w = w - lr * D_w  # Update w
    b = b - lr * D_b  # Update b
    print(w)
    print(b)

print(w, b)

# Making predictions
y_hat = w * x + b
plt.scatter(x, y)
plt.plot([min(x), max(x)], [min(y_hat), max(y_hat)], color='red')
plt.show()