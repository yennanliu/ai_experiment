import requests
from bs4 import BeautifulSoup

url = "https://en.wikipedia.org/wiki/Palantir_Technologies"
headers = {'User-Agent': 'PalantirScraper/1.0'}
response = requests.get(url, headers=headers)
soup = BeautifulSoup(response.text, 'html.parser')

print("--- Spans with IDs ---")
for span in soup.find_all('span', id=True):
    if 'history' in span['id'].lower() or 'product' in span['id'].lower():
        print(f"ID: {span['id']}, Name: {span.name}")

print("\n--- H2 Headings ---")
for h2 in soup.find_all('h2'):
    print(f"H2 Text: {h2.get_text(strip=True)}")
