import requests, os, random, json
from dotenv import load_dotenv

load_dotenv("secret.env")

# tmdb api auth
def authenticate():
    url = "https://api.themoviedb.org/3/authentication"

    headers = {"accept": "application/json",
            "Authorization": f"Bearer {os.getenv('TMDB_API_READ')}"}

    response = requests.get(url, headers=headers)

    if response.status_code == 200:
        print("Authentication successful")
    else:
        print("Authentication failed")
        print(response.json().get("status_message"))


# tmdb api session
def create_session():
    url = "https://api.themoviedb.org/3/authentication/token/new"

    headers = {
        "accept": "application/json",
        "Authorization": f"Bearer {os.getenv('TMDB_API_READ')}"
    }

    response = requests.get(url, headers=headers)

    print(response.text)

    if response.status_code == 200:
        print("Session created")
        return response.json().get("request_token")
    else:
        print("Session creation failed")
        print(response.json().get("status_message"))


# for loop for getting the movie details for each genre
def get_movie_details(genre_ids, genre_names):
    for i in range(len(genre_ids)):
        counter = 0
        movielist = get_movielist(genre_ids[i])

        # create a folder to store the movie list
        if not os.path.exists("movielists"):
            os.makedirs("movielists")

        print(f"Downloading posters and writing movie list for {genre_names[i]} with {len(movielist)} movies")

        # create a folder to store the posters
        if not os.path.exists("posters"):
            os.makedirs("posters")

        # create a folder for each genre inside the 'posters' folder
        if not os.path.exists(f"posters/{genre_names[i]}"):
            os.makedirs(f"posters/{genre_names[i]}")

        for movie in movielist:
            # store the movie id, title, poster path, release date in a file")
            if download_poster(movie['movie_poster'], movie['movie_id'], movie['movie_title'], genre_names[i], counter):
                with open("movielists/master.out", "a") as file:
                    file.write(f"{movie['movie_id']} - {movie['movie_title']} - {movie['movie_poster']} - {genre_names[i]}\n")
                    counter += 1

            with open(f"movielists/{genre_names[i]}.out", "a") as file:
                file.write(f"{movie['movie_id']} - {movie['movie_title']} - {movie['movie_poster']}\n")

        print(f"Done downloading posters and writing movie list for {genre_names[i]}")

# gets the movie list for each genre
def get_movielist(genre_id):
    counter = 0
    page_list = []
    movie_list = []
    page_number = 1
    while counter < 2000:
        url = f"https://api.themoviedb.org/3/discover/movie?include_adult=true&include_video=false&page={str(page_number)}&sort_by=popularity.desc&with_genres={genre_id}"

        headers = {
            "accept": "application/json",
            "Authorization": f"Bearer {os.getenv('TMDB_API_READ')}"
        }

        response = requests.get(url, headers=headers)

        if response.status_code == 200:
            print(f"Movie list retrieved for genre {genre_id} page {page_number} counter {counter}")
            for movie in response.json().get("results"):
                if movie.get("poster_path") is None:
                    print(f"Movie {movie.get('id')} - {movie.get('title')} has no poster")

                else:
                    movie_list.append({"movie_id": movie["id"], "movie_title": movie["title"], "movie_poster": movie["poster_path"]})
                    counter += 1
                    if counter == 2000:
                        break

            page_list.append(page_number)
            page_number = get_random_page_number(page_list)

        else:
            print("Movie list retrieval failed")
            print(response.json().get("status_message"))
    return movie_list

# get a random page number for getting the movie list
def get_random_page_number(page_list):
    while True:
        page_number = random.randint(1, 500)
        if page_number not in page_list:
            return page_number 

# count the number of movies in the object
def count_movies_in_list_response(response):
    counter = 0
    for movie in response.get("results"):
        if movie.get("poster_path") is not None:
            counter += 1
        else:
            print(f"Movie {movie.get('id')} - {movie.get('title')} has no poster")
    return counter


# download the poster of the movie
def download_poster(poster_path, movie_id, movie_title, genre_name, counter):
    # download the poster to the movie id for easy reference

    print(f"{counter}/2000  Downloading poster for movie {movie_id} - {movie_title}")
    url = f"https://image.tmdb.org/t/p/original{poster_path}"

    headers = {
        "accept": "application/json",
        "Authorization": f"Bearer {os.getenv('TMDB_API_READ')}"
    }

    response = requests.get(url, headers=headers)

    if response.status_code == 200:
        with open(f"posters/{genre_name}/{movie_id}.jpg", "wb") as file:
            file.write(response.content)
        return True
    
    else:
        print(f"Failed to download poster for movie {movie_id}")
        return False
        

# main function
def main():
    authenticate()
    token = create_session()
    
    genre_ids = [
        28, # Action
        12, # Adventure
        16, # Animation
        35, # Comedy
        80, # Crime
        18, # Drama
        14, # Fantasy
        10749, # Romance
        53, # Thriller
    ]

    genre_names = [
        "Action",
        "Adventure",
        "Animation",
        "Comedy",
        "Crime",
        "Drama",
        "Fantasy",
        "Romance",
        "Thriller"
    ]

    get_movie_details(genre_ids, genre_names)

    

if __name__ == "__main__":
    main()

# TODO:
# create a master file
# create a separate file with all the id and genre and title
# create a file for id and poster