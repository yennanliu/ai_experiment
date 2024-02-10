import requests
from bs4 import BeautifulSoup

# URL of the website
url = "https://www.twblogs.net/c/5b7a8bb52b7177392c9643a1/"

# Send a GET request to the URL
response = requests.get(url)

# Check if the request was successful (status code 200)
if response.status_code == 200:
    # Parse the HTML content of the response
    soup = BeautifulSoup(response.content, "html.parser")
    print("soup = " + str(soup))

    elements = soup.find_all("a", class_="", attrs={"data-post-id": True})

    # Extract and print the text content of each element
    for element in elements:
        print(element.text.strip())

else:
    print("Failed to retrieve data from the website")
