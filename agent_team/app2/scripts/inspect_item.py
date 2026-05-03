import requests
from bs4 import BeautifulSoup

def inspect_item():
    search_url = "https://www.amazon.com/s?k=ps5&crid=2IJH0FK0RJAN8&sprefix=ps5%2Caps%2C364&ref=nb_sb_noss_1"
    headers = {
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        "Accept-Language": "en-US,en;q=0.9",
    }
    response = requests.get(search_url, headers=headers)
    soup = BeautifulSoup(response.text, 'html.parser')
    items = soup.find_all('div', {'data-component-type': 's-search-result'})
    if items:
        with open("item_debug.html", "w") as f:
            f.write(items[0].prettify())
        print("Wrote item_debug.html")

if __name__ == "__main__":
    inspect_item()
