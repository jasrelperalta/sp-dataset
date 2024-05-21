import pandas as pd


csv_file_path = 'movie_dataset_combined2.csv'

lendata = pd.read_csv(csv_file_path)

print(len(lendata))

total = 0

# create a new csv file containing the movie posters and their genres retaining the 60% training data by skipping the 30% validation data and 10% test data
data = pd.read_csv(csv_file_path, skiprows=[i+1 for i in range(len(lendata)) if (i % 10 > 6) or (i % 10 == 0)])

data.to_csv('movie_dataset_train2.csv', index=False)

print(len(data), "training data")
total = total + len(data)

# create a csv file containing the movie posters and their genres retaining the 30% validation data

data = pd.read_csv(csv_file_path, skiprows=[i+1 for i in range(len(lendata)) if (i % 10 < 7) or (i % 10 == 0)])

data.to_csv('movie_dataset_val2.csv', index=False)

print(len(data), "validation data")
total = total + len(data)

# create a csv file containing the movie posters and their genres retaining the 10% test data

data = pd.read_csv(csv_file_path, skiprows=[i+1 for i in range(len(lendata)) if i % 10 != 0])

data.to_csv('movie_dataset_test2.csv', index=False)

print(len(data), "test data")
total = total + len(data)

print("CSV files created successfully")
print("Total data:", total)