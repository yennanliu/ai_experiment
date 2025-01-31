import requests
from bs4 import BeautifulSoup
import re

# URL of the YouTube Taiwan homepage
url = "https://www.youtube.com/"

# Send a GET request to the URL
response = requests.get(url)

# Parse the HTML content of the response
soup = BeautifulSoup(response.content, "html.parser")

# Find all anchor tags with 'href' attribute containing '/watch?v=' (YouTube video links)
video_links = soup.find_all("a", href=re.compile(r"/watch\?v="))

# Extract the video IDs from the href attributes
video_ids = [link["href"].split("/watch?v=")[1].split("&")[0] for link in video_links]

# Print the collected video IDs
for video_id in video_ids:
    print(video_id)
