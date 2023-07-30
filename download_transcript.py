import sys
import os
import requests
from bs4 import BeautifulSoup
import re

def download_transcript(url, save_file_path):
    try:
        response = requests.get(url)
        if response.status_code == 200:
            with open(save_file_path + ".html", 'wb') as file:
                file.write(response.content)
            print(f"Transcript downloaded successfully and saved as '{save_file_path}.html'.")

            # Remove HTML tags and save as a text file
            with open(save_file_path + ".html", 'r', encoding='utf-8') as html_file:
                html_content = html_file.read()
                soup = BeautifulSoup(html_content, 'html.parser')
                text_content = soup.get_text()

                # Remove all blank new lines from the content
                text_content = re.sub(r'\n\s*\n', '\n', text_content)

                # Remove time stamps in the format (00:18:20)
                text_content = re.sub(r'\(\d+:\d+:\d+\)\s*', '', text_content)

                with open(save_file_path + ".txt", 'w', encoding='utf-8') as txt_file:
                    txt_file.write(text_content)
                print(f"Text content saved as '{save_file_path}.txt'.")
        else:
            print(f"Failed to download the transcript. Status code: {response.status_code}")
    except requests.exceptions.RequestException as e:
        print(f"An error occurred while downloading the transcript: {e}")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python download_transcript.py <url>")
        sys.exit(1)

    url = sys.argv[1]
    filename = url.replace("://", "").replace("/", "")
    transcript_dir = "/Users/gauta/Downloads/podcasts/transcripts"
    save_file_path = os.path.join(transcript_dir, filename)
    download_transcript(url, save_file_path)

