import requests
from bs4 import BeautifulSoup

url = "https://en.wikipedia.org/wiki/Palantir_Technologies"
headers = {'User-Agent': 'PalantirScraper/1.0'}
response = requests.get(url, headers=headers)
soup = BeautifulSoup(response.text, 'html.parser')

history_h2 = soup.find('h2', id='History')
if history_h2:
    print(f"Parent of History H2: {history_h2.parent.name}, classes={history_h2.parent.get('class')}")
    print(f"Siblings of History H2: {[s.name for s in history_h2.find_next_siblings()]}")
    
    # Check if the content is actually in the next sibling of the parent or something
    # Actually, let's see the first 1000 chars of the parent's HTML
    # print(history_h2.parent.prettify()[:1000])
else:
    print("History H2 with ID 'History' not found")
