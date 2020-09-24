# create and evaluate an updated autoregressive model
import csv
import re
import numpy as np
import pandas as pd

from matplotlib import pyplot as plt
from statsmodels.tsa.ar_model import AutoReg
from sklearn.metrics import mean_squared_error
from math import sqrt


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
Z = pd.DataFrame(fileopen)[2]  # temperature
str_to_float(Z, temperature)
X = np.array(temperature)

# split dataset
train, test = X[1:len(X)//2], X[len(X)//2:]

# train autoregression
window = 250
model = AutoReg(train, lags=250)
model_fit = model.fit()
coef = model_fit.params

# walk forward over time steps in test
history = train[len(train) - window:]
history = [history[i] for i in range(len(history))]
predictions = list()

for t in range(len(test)):
    length = len(history)
    lag = [history[i] for i in range(length - window, length)]
    y_hat = coef[0]
    for d in range(window):
        y_hat += coef[d + 1] * lag[window - d - 1]
    obs = test[t]
    predictions.append(y_hat)
    history.append(obs)
    print('predicted=%f,\t expected=%f' % (y_hat, obs))
rmse = sqrt(mean_squared_error(test, predictions))
print('Test RMSE: %.3f' % rmse)

# plot
model_fit.plot_predict(start=len(train), end=len(train) + len(X), dynamic=False)
plt.plot(X, label='original')

plt.plot(predictions, color='red', linewidth=1, label='prediction')
plt.xlabel("Entry")
plt.ylabel("Temperature")
plt.legend()
plt.xlim(-100, 2500)
plt.ylim(10, 40)
plt.show()
