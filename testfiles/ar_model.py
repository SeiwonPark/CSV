# create and evaluate a static autoregressive model
import csv
import re
import numpy as np
import pandas as pd
from pandas import read_csv
from matplotlib import pyplot as plt
from statsmodels.tsa.ar_model import AutoReg
from sklearn.metrics import mean_squared_error
from math import sqrt


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
Z = pd.DataFrame(fileopen)[2]  # temperature
str_to_float(Z, temperature)
X = np.array(temperature)
plt.plot(X)
plt.show()
plt.close()

train, test = X[1:len(X)//2], X[len(X)//2:]

# train autoregression
model = AutoReg(train, lags=250)
model_fit = model.fit()
print('Coefficients: %s' % model_fit.params)

# make predictions
predictions = model_fit.predict(start=len(train), end=len(train) + len(test) - 1, dynamic=False)

for i in range(len(predictions)):
    print('predicted=%f,\t expected=%f' % (predictions[i], test[i]))

rmse = sqrt(mean_squared_error(test, predictions))
print('Test RMSE: %.3f' % rmse)


# plot results
model_fit.plot_predict(start=len(train), end=len(train) + len(test) + 200, dynamic=False)

plt.plot(test, color='black')
plt.plot(predictions, color='magenta')
plt.show()
