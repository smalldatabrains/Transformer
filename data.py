import requests
from bs4 import BeautifulSoup
import os

class GeniusLyrics:
    def __init__(self, api_key):
        self.base_url = "https://api.genius.com"
        self.headers = {'Authorization': f'Bearer {api_key}'}

    def search_artist(self, artist_name):
        search_url = f"{self.base_url}/search"
        data = {'q': artist_name}
        response = requests.get(search_url, headers=self.headers, params=data, verify=False)
        if response.status_code == 200:
            return response.json()
        else:
            return None

    def get_artist_id(self, artist_name):
        artist_data = self.search_artist(artist_name)
        print(artist_data)
        if artist_data:
            for hit in artist_data['response']['hits']:
                if hit['result']['primary_artist']['name'].lower() == artist_name.lower():
                    return hit['result']['primary_artist']['id']
        return None

    def get_artist_songs(self, artist_id, page=1):
        songs = []
        while True:
            url = f"{self.base_url}/artists/{artist_id}/songs"
            params = {'page': page, 'per_page': 50}
            response = requests.get(url, headers=self.headers, params=params, verify=False)
            if response.status_code != 200:
                break
            song_data = response.json()
            songs.extend(song_data['response']['songs'])
            if not song_data['response']['next_page']:
                break
            page += 1
        return songs

    def scrape_lyrics(self, song_url):
        response = requests.get(song_url)
        if response.status_code == 200:
            soup = BeautifulSoup(response.content, 'html.parser')
            lyrics_div = soup.find('div', class_='lyrics')
            if lyrics_div:
                return lyrics_div.get_text()
            else:
                lyrics_div = soup.find('div', class_='SongPage__lyrics')
                if lyrics_div:
                    return lyrics_div.get_text()
        return None

    def save_lyrics_to_file(self, song_title, lyrics, artist_name):
        # Create directory for the artist if it doesn't exist
        directory = f"{artist_name}_lyrics"
        if not os.path.exists(directory):
            os.makedirs(directory)
        
        # Clean song title to make it filename-safe
        safe_song_title = song_title.replace('/', '_').replace('\\', '_')
        filename = f"{directory}/{safe_song_title}.txt"
        
        with open(filename, 'w', encoding='utf-8') as file:
            file.write(lyrics)
    
    def get_all_lyrics_by_artist(self, artist_name):
        artist_id = self.get_artist_id(artist_name)
        if not artist_id:
            return "Artist not found."
        
        songs = self.get_artist_songs(artist_id)
        
        for song in songs:
            song_title = song['title']
            song_url = song['url']
            lyrics = self.scrape_lyrics(song_url)
            if lyrics:
                self.save_lyrics_to_file(song_title, lyrics, artist_name)
        
        return "Lyrics have been saved."

# Example usage:
if __name__ == "__main__":
    api_key = ""  # Replace with your Genius API key
    genius = GeniusLyrics(api_key)
    artist_name = "Damso"  # Example artist name
    result = genius.get_all_lyrics_by_artist(artist_name)
    print(result)
