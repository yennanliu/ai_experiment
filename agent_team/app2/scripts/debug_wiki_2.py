import requests
from bs4 import BeautifulSoup

url = "https://en.wikipedia.org/wiki/Palantir_Technologies"
headers = {'User-Agent': 'PalantirScraper/1.0'}
response = requests.get(url, headers=headers)
soup = BeautifulSoup(response.text, 'html.parser')

history_h2 = None
for h2 in soup.find_all('h2'):
    if 'History' in h2.get_text():
        history_h2 = h2
        break

if history_h2:
    print(f"Found History H2: {history_h2}")
    print("Next 5 siblings:")
    for i, sib in enumerate(history_h2.find_next_siblings()[:5]):
        print(f"{i}: Name={sib.name}, Classes={sib.get('class')}")
else:
    print("History H2 NOT found")
