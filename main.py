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

def get_movielist():
    url = "https://api.themoviedb.org/3/discover/movie?include_adult=true&include_video=false&language=en-US&page=1"

    headers = {
        "accept": "application/json",
        "Authorization": f"Bearer {os.getenv('TMDB_API_READ')}"
    }

    response = requests.get(url, headers=headers)

    if response.status_code == 200:
        print("Movie list retrieved")
        return response.json()
    else:
        print("Movie list retrieval failed")
        print(response.json().get("status_message"))

def download_poster(poster_path, movie_id, movie_title):
    # download the poster to the movie id for easy reference
    # create a folder to store the posters
    if not os.path.exists("posters"):
        os.makedirs("posters")

    url = f"https://image.tmdb.org/t/p/original{poster_path}"
    response = requests.get(url)

    if response.status_code == 200:
        print(f"Downloading poster for movie {movie_title}")
        with open(f"posters/{movie_id}-{movie_title}.jpg", "wb") as file:
            file.write(response.content)
    else:
        print(f"Failed to download poster for movie {movie_id}")
        print(response.json().get("status_message"))


def main():
    authenticate()
    token = create_session()
    movielist = get_movielist()
    
    # create a file to store the movie list cleanly by writing to a file
    with open("movielist.out", "w") as file:
        for movie in movielist.get("results"):
            # get title and link to poster
            title = movie.get("title")
            poster = movie.get("poster_path")
            genres = movie.get("genre_ids")
            release_date = movie.get("release_date")
            movie_id = movie.get("id")
            file.write(f"{movie_id} - {title} - {release_date} - {genres} - {poster}\n")
            
    # download the posters of the movies in the list

    for movie in movielist.get("results"):
        download_poster(movie.get("poster_path"), movie.get("id"), movie.get("title"))

    print("Done downloading posters")


if __name__ == "__main__":
    main()
