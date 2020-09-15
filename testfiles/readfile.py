line_counter = 0
data_header = []
index_list = []

with open("feeds.csv") as index_data:

    while True:
        data = index_data.readline()
        if not data:
            break
        if line_counter == 0:
            data_header = data.split(",")
        else:
            index_list.append(data.split(","))
        line_counter += 1

print("Headers: ", data_header)
for i in range(1, line_counter):
    print("Data", i, ":", index_list[i-1])
print(len(index_list))