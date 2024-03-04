import requests, os
from dotenv import load_dotenv

load_dotenv("secret.env")

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

def get_movielist(genre_id):
    counter = 0
    page_counter = 1
    movie_list = []
    while counter < 2001:
        url = f"https://api.themoviedb.org/3/discover/movie?include_adult=true&include_video=false&page={str(page_counter)}&sort_by=popularity.desc&with_genres={genre_id}"

        headers = {
            "accept": "application/json",
            "Authorization": f"Bearer {os.getenv('TMDB_API_READ')}"
        }

        response = requests.get(url, headers=headers)

        if response.status_code == 200:
            print(f"Movie list retrieved for genre {genre_id} page {page_counter} counter {counter}")
            counter += count_movies_in_list_response(response.json())
            page_counter += 1
            movie_list.append(response.json().get("results"))

        else:
            print("Movie list retrieval failed")
            print(response.json().get("status_message"))

    return movie_list

def count_movies_in_list_response(response):
    return len(response.get("results"))

def download_poster(poster_path, movie_id, movie_title, genre_name):
    # download the poster to the movie id for easy reference
    # create a folder to store the posters
    print(f"Downloading poster for movie {movie_id} - {movie_title}")
    if not os.path.exists("posters"):
        os.makedirs("posters")

    # create a folder for each genre inside the 'posters' folder
    if not os.path.exists(f"posters/{genre_name}"):
        os.makedirs(f"posters/{genre_name}")

    url = f"https://image.tmdb.org/t/p/original{poster_path}"

    headers = {
        "accept": "application/json",
        "Authorization": f"Bearer {os.getenv('TMDB_API_READ')}"
    }

    response = requests.get(url, headers=headers)

    if response.status_code == 200:
        with open(f"posters/{genre_name}/{movie_id}.jpg", "wb") as file:
            file.write(response.content)
    else:
        print(f"Failed to download poster for movie {movie_id}")


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

    for i in range(len(genre_ids)):
        movielist = get_movielist(genre_ids[i])

        # create a folder to store the movie list
        if not os.path.exists("movielists"):
            os.makedirs("movielists")

        with open(f"movielists/{genre_names[i]}.out", "w") as file:
            print(f"Writing movie list for {genre_names[i]}")
            for page in movielist:
                for movie in page:
                    # store the movie id, title, poster path, release date in a file
                    file.write(f"{movie.get('id')} - {movie.get('title')} - {movie.get('poster_path')} - {movie.get('release_date')}\n")
            print(f"Done writing movie list for {genre_names[i]}")

        # download the posters of the movies in the list
        # create a separate folder to store the posters per genre
        for page in movielist:
            for movie in page:
                download_poster(movie.get("poster_path"), movie.get("id"), movie.get("title"), genre_names[i])

        print(f"Done downloading posters for {genre_names[i]}")

if __name__ == "__main__":
    main()
