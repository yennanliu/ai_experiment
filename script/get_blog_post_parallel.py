from selenium import webdriver
from selenium.webdriver.common.by import By
import time
from bs4 import BeautifulSoup
from multiprocessing import Process, Manager

def collect_urls(url, shared_urls):
    # Initialize Chrome webdriver
    driver = webdriver.Chrome()

    # Open the webpage
    driver.get(url)

    # Wait for the page to load completely (adjust the sleep time as needed)
    time.sleep(5)

    # Scroll down to the bottom of the page and collect URLs
    urls = []
    while len(urls) < 50:
        # Scroll down using JavaScript
        driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
        
        # Wait for some time to let the content load
        time.sleep(2)

        # Get the page source
        html = driver.page_source

        # Parse the HTML using BeautifulSoup
        soup = BeautifulSoup(html, 'html.parser')

        # Find all anchor tags with href containing '/a/'
        elements = soup.find_all('a', href=lambda href: href and '/a/' in href)

        for element in elements:
            url = element['href']
            if url:  # Ensure that the URL is not empty
                urls.append(url)
        
        # Break the loop if no more content is loaded
        if driver.execute_script("return window.innerHeight + window.scrollY") >= driver.execute_script("return document.body.scrollHeight"):
            break

    # Add unique URLs to the shared set
    for url in urls:
        shared_urls.append(url)

    # Close the webdriver
    driver.quit()

if __name__ == "__main__":
    # URL of the webpage to scroll down
    url = "https://www.twblogs.net/c/5b7a8bb62b7177392c9643c0/"

    # Create a shared set to store the URLs
    manager = Manager()
    shared_urls = manager.list()

    # Create multiple processes to collect URLs in parallel
    processes = []
    PARALLEL = 2

    for _ in range(PARALLEL):  # You can adjust the number of processes
        p = Process(target=collect_urls, args=(url, shared_urls))
        p.start()
        processes.append(p)

    # Wait for all processes to finish
    for p in processes:
        p.join()

    # Save the list of unique URLs to a text file
    # with open('urls.txt', 'w') as f:
    #     for url in shared_urls:
    #         f.write(url + '\n')
    print ("length of url = " + str(len(shared_urls)))
    print (shared_urls)
