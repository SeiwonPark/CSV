import csv
import re
import pandas as pd
import numpy
from matplotlib import pyplot as plt
from statsmodels.tsa.ar_model import AutoReg
from sklearn.metrics import mean_squared_error
from math import sqrt


# create a difference transform of the dataset
# def difference(dataset):
#     diff = list()
#     for i in range(1, len(dataset)):
#         value = dataset[i] - dataset[i - 1]
#         diff.append(value)
#     return numpy.array(diff)


# Make a prediction give regression coefficients and lag obs
def predict(coef, history):
    y_hat = coef[0]
    for i in range(1, len(coef)):
        y_hat += coef[i] * history[-i]
    return y_hat


def opencsv(filename):
    f = open(filename, 'r')
    reader = csv.reader(f)
    output = list()
    for i in reader:
        output.append(i)
    return numpy.array(output)


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
X = numpy.array(temperature)

# split dataset
size = int(len(X) * 0.66)
train, test = X[1:size], X[size:]

# train autoregression
model = AutoReg(train, lags=6)
model_fit = model.fit()
coef = model_fit.params

# walk forward over time steps in test
history = [train[i] for i in range(len(train))]
predictions = list()

for t in range(len(test)):
    y_hat = predict(coef, history)
    obs = test[t]
    predictions.append(y_hat)
    history.append(obs)
    print("Test {}".format(t+1))
    print('predicted = %f,\t expected = %f' % (predictions[t], test[t]))

rmse = sqrt(mean_squared_error(test, predictions))
print('Test RMSE: %.3f' % rmse)

model_fit.plot_predict(1, 1500)
plt.plot(test, color='black')
plt.plot(predictions, color='red')
plt.xlabel("Entry")
plt.ylabel("Temperature")
plt.legend()
plt.xlim(-100, 2500)
plt.ylim(10, 40)
plt.show()
