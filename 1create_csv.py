import csv, os

# Python script to create a CSV file
# This script is used to create a CSV file from the movie details that we have collected to be used as a dataset

# Function to create a CSV file

def create_csv():
    # Field names
    fields = ['filename', 'Action', 'Adventure', 'Animation', 'Comedy', 'Crime', 'Drama', 'Fantasy', 'Romance', 'Thriller']

    # Data rows of csv file
    # iterate over the whole folder per genre and create a row for each image
    data = []
    for genre in os.listdir("posters2"):
        for image in os.listdir(f"posters2/{genre}"):
            print(f"Processing {image} from {genre}")
            # split the filename to get the movie id
            image = image.split("_")[1]
            row = [image]
            # prefill the row with 0s
            for i in range(9):
                row.append(0)
            # set the genre of the image to 1
            row[fields.index(genre)] = 1
            data.append(row)

    # Name of the CSV file
    filename = "movie_dataset2.csv"

    # Write to CSV file
    with open(filename, 'w') as csvfile:
        # Create a CSV writer object
        csvwriter = csv.writer(csvfile)

        # Write the fields
        csvwriter.writerow(fields)

        # Write the data rows
        csvwriter.writerows(data)

    print(f"CSV file {filename} created successfully")

# Function that combines the rows with the same movie id
def combine_rows():
    # Read the CSV file
    with open("movie_dataset2.csv", "r") as file:
        reader = csv.reader(file)
        rows = list(reader)

    # Dictionary to store the combined rows
    combined_rows = {}

    # Iterate over the rows
    for row in rows[1:]:
        movie_id = row[0]
        if movie_id in combined_rows:
            # Combine the rows
            combined_rows[movie_id] = [int(a) or int(b) for a, b in zip(combined_rows[movie_id], row[1:])]
        else:
            combined_rows[movie_id] = [int(x) for x in row[1:]]

    # Write the combined rows to a new CSV file
    with open("movie_dataset_combined2.csv", "w") as file:
        writer = csv.writer(file)
        writer.writerow(rows[0])
        writer.writerows([[k] + v for k, v in combined_rows.items()])

    print("Combined CSV file created successfully")

# Call the function to create the CSV file
create_csv()

# Call the function to combine the rows
combine_rows()

