import numpy as np
import pandas as pd
import csv
import re
from matplotlib import pyplot as plt
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.arima_model import ARIMA, ARIMA_DEPRECATION_WARN
from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()
import warnings
warnings.filterwarnings('ignore', 'statsmodels.tsa.arima_model.ARMA', FutureWarning)
warnings.filterwarnings('ignore', 'statsmodels.tsa.arima_model.ARIMA', FutureWarning)
warnings.warn(ARIMA_DEPRECATION_WARN, FutureWarning)

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


fileopen = opencsv('feeds.csv')
temperature = list()
str_data = pd.DataFrame(fileopen)[2]
str_to_float(str_data, temperature)
W = np.array(temperature)

col = ['Temperature']
X = pd.DataFrame(W.reshape(len(W), 1), columns=col)
print(X)

plt.xlabel('Entry')
plt.ylabel('Temperature')
plt.plot(X)
plt.show()
plt.close()

rolling_mean = X.rolling(window = 12).mean()
rolling_std = X.rolling(window = 12).std()
plt.plot(X, color = 'blue', label = 'Original')
plt.plot(rolling_mean, color = 'red', label = 'Rolling Mean')
plt.plot(rolling_std, color = 'black', label = 'Rolling Std')
plt.legend(loc = 'best')
plt.title('Rolling Mean & Rolling Standard Deviation')

plt.show()
plt.close()

result = adfuller(X['Temperature'])
print('ADF Statistic: {}'.format(result[0]))
print('p-value: {}'.format(result[1]))
print('Critical Values:')
for key, value in result[4].items():
    print('\t{}: {}'.format(key, value))

X_log = np.log(X)
plt.plot(X_log)
plt.show()
plt.close()


def get_stationarity(timeseries):
    # rolling statistics
    rolling_mean = timeseries.rolling(window=12).mean()
    rolling_std = timeseries.rolling(window=12).std()

    # rolling statistics plot
    original = plt.plot(timeseries, color='blue', label='Original')
    mean = plt.plot(rolling_mean, color='red', label='Rolling Mean')
    std = plt.plot(rolling_std, color='black', label='Rolling Std')
    plt.legend(loc='best')
    plt.title('Rolling Mean & Standard Deviation')
    plt.show(block=False)

    # Dickey–Fuller test:
    result = adfuller(timeseries['Temperature'])
    print('ADF Statistic: {}'.format(result[0]))
    print('p-value: {}'.format(result[1]))
    print('Critical Values:')
    for key, value in result[4].items():
        print('\t{}: {}'.format(key, value))

rolling_mean = X_log.rolling(window=12).mean()
X_log_minus_mean = X_log - rolling_mean
X_log_minus_mean.dropna(inplace=True)
get_stationarity(X_log_minus_mean)
plt.close()

rolling_mean_exp_decay = X_log.ewm(halflife=12, min_periods=0, adjust=True).mean()
X_log_exp_decay = X_log - rolling_mean_exp_decay
X_log_exp_decay.dropna(inplace=True)
get_stationarity(X_log_exp_decay)
plt.close()

X_log_shift = X_log - X_log.shift()
X_log_shift.dropna(inplace=True)
get_stationarity(X_log_shift)
plt.close()

# decomposition = seasonal_decompose(X_log)  ####
model = ARIMA(X_log, order=(2,1,2))
results = model.fit(disp=-1)
# plt.plot(X_log_shift)
# plt.plot(results.fittedvalues, color='red')
# plt.show()
# plt.close()

predictions_ARIMA_diff = pd.Series(results.fittedvalues, copy=True)
predictions_ARIMA_diff_cumsum = predictions_ARIMA_diff.cumsum()
predictions_ARIMA_log = pd.Series(X_log['Temperature'].iloc[0], index=X_log.index)
predictions_ARIMA_log = predictions_ARIMA_log.add(predictions_ARIMA_diff_cumsum, fill_value=0)
predictions_ARIMA = np.exp(predictions_ARIMA_log)
plt.plot(X)
plt.plot(predictions_ARIMA)
results.plot_predict(1,2500)
plt.show()
plt.close()

print("Hi")

