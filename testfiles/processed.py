import pandas as pd
import csv
import re


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

print(temperature)
print(humidity)
print(pressure)
print(pm1_0)
print(pm2_5)
print(pm10)
