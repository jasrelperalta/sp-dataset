import pandas as pd


csv_file_path = 'movie_dataset_combined.csv'

lendata = pd.read_csv(csv_file_path)

print(len(lendata))

# create a new csv file containing the movie posters and their genres retaining the 60% training data by skipping the 30% validation data and 10% test data
data = pd.read_csv(csv_file_path, skiprows=[i+1 for i in range(len(lendata)) if i % 10 < 6])

data.to_csv('movie_dataset_train2.csv', index=False)

# create a csv file containing the movie posters and their genres retaining the 30% validation data

data = pd.read_csv(csv_file_path, skiprows=[i+1 for i in range(len(lendata)) if i % 10 > 6])

print(len(data))

data.to_csv('movie_dataset_val2.csv', index=False)

# create a csv file containing the movie posters and their genres retaining the 10% test data

data = pd.read_csv(csv_file_path, skiprows=[i+1 for i in range(len(lendata)) if i % 10 == 0])

print(len(data))

data.to_csv('movie_dataset_test2.csv', index=False)

print("CSV files created successfully")