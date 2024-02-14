import requests
from bs4 import BeautifulSoup
from datetime import datetime, timedelta

from selenium import webdriver
from selenium.webdriver.common.keys import Keys
# https://blog.csdn.net/m0_49076971/article/details/126233151

from selenium.webdriver.common.by import By
import time


from selenium import webdriver
from selenium.webdriver.common.keys import Keys
import time

# URL of the webpage to scroll down
url = "https://www.twblogs.net/c/5b7a8bb62b7177392c9643c0/"

# Path to chromedriver executable
#chromedriver_path = "/usr/local/bin/chromedriver"

# Initialize Chrome webdriver
driver = webdriver.Chrome()

# Open the webpage
driver.get(url)

# Wait for the page to load completely (adjust the sleep time as needed)
time.sleep(5)

# Scroll down to the bottom of the page and collect URLs
urls = []
while len(urls) < 300:
    # Scroll down using JavaScript
    driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
    
    # Wait for some time to let the content load
    time.sleep(2)

    html = driver.page_source

    soup = BeautifulSoup(html, 'html.parser')

    #print (">>> soup = " + str(soup))

    
    # Find all anchor tags with href containing '/a/'
    elements = soup.find_all('a', href=lambda href: href and '/a/' in href)

    for element in elements:
        url = element['href']
        if url:  # Ensure that the URL is not empty
            urls.append(url)

    # Break the loop if no more content is loaded
    if driver.execute_script("return window.innerHeight + window.scrollY") >= driver.execute_script("return document.body.scrollHeight"):
        break

# Print the list of URLs
for i, url in enumerate(urls, 1):
    print(f"{i}. {url}")

# Close the webdriver
driver.quit()


print (len(urls))
print ("------>")
print (urls)
print ("------>")