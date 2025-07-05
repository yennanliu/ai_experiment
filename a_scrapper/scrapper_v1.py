import requests
from bs4 import BeautifulSoup

# URL to scrape
url = ''

# Set a user-agent to avoid basic bot blocking
headers = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'
}

# Send GET request
response = requests.get(url, headers=headers)

# Check response status
if response.status_code == 200:
    soup = BeautifulSoup(response.content, 'html.parser')

    # Find all <video> tags or <a> tags that might contain video URLs
    video_links = []

    # Option 1: Look for <video> tags
    for video_tag in soup.find_all('video'):
        src = video_tag.get('src')
        if src:
            video_links.append(src)

    # Option 2: Look for links to video files (e.g. .mp4)
    for a_tag in soup.find_all('a', href=True):
        href = a_tag['href']
        if any(ext in href for ext in ['.mp4', '.m3u8']):
            video_links.append(href)

    # Remove duplicates
    video_links = list(set(video_links))

    # Print results
    print("Found video URLs:")
    for link in video_links:
        print(link)
else:
    print(f"Failed to fetch page. Status code: {response.status_code}")