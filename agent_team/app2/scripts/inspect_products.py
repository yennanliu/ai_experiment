import requests
from bs4 import BeautifulSoup

url = "https://en.wikipedia.org/wiki/Palantir_Technologies"
headers = {'User-Agent': 'PalantirScraper/1.0'}
response = requests.get(url, headers=headers)
soup = BeautifulSoup(response.text, 'html.parser')

products_h2 = None
for h2 in soup.find_all('h2'):
    if 'Products' in h2.get_text():
        if h2.parent and 'mw-heading' in h2.parent.get('class', []):
            products_h2 = h2.parent
        else:
            products_h2 = h2
        break

if products_h2:
    print(f"Products section starts with: {products_h2.get_text()}")
    for sib in products_h2.find_next_siblings():
        if sib.name == 'h2' or (sib.name == 'div' and 'mw-heading2' in sib.get('class', [])):
            print(f"Section ends at: {sib.get_text()}")
            break
        print(f"<{sib.name}> {sib.get_text()[:100].strip()}")
else:
    print("Products H2 not found")
