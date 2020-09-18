# line_counter = 0
# data_header = []
# index_list = []
#
# with open("https://api.thingspeak.com/channels/1136756/fields/2.csv") as index_data:
#     while True:
#         data = index_data.readline()
#         if not data:
#             break
#         if line_counter == 0:
#             data_header = data.split(",")
#         else:
#             index_list.append(data.split(","))
#             line_counter += 1
#
# print("Headers: ", data_header)
# for i in range(1, line_counter):
#     print("Data", i, ":", index_list[i-1])
#     print(len(index_list))


import pandas as pd

data = pd.read_csv(r'feeds.csv')    # read csv file

df = pd.DataFrame(data, columns=['field1', 'field2', 'field3', 'field4', 'field5', 'field6'])

print(df)
