from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException
import time
import json
import os
from datetime import datetime

class LinkedInJobScraper:
    def __init__(self, email, password):
        self.email = email
        self.password = password
        self.driver = None
        self.jobs_data = []

    def setup_driver(self):
        try:
            options = webdriver.ChromeOptions()
            options.add_argument('--headless')  # Run in headless mode
            options.add_argument('--no-sandbox')
            options.add_argument('--disable-dev-shm-usage')
            # Add these options to help with SSL issues
            options.add_argument('--ignore-certificate-errors')
            options.add_argument('--ignore-ssl-errors')
            # Disable logging
            options.add_experimental_option('excludeSwitches', ['enable-logging'])
            
            # Add user agent to make the request more browser-like
            options.add_argument('--user-agent=Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36')
            
            self.driver = webdriver.Chrome(options=options)
            self.driver.set_page_load_timeout(30)  # Set page load timeout
        except Exception as e:
            print(f"Error setting up driver: {str(e)}")
            raise

    def login(self):
        try:
            self.driver.get('https://www.linkedin.com/login')
            WebDriverWait(self.driver, 10).until(
                EC.presence_of_element_located((By.ID, "username"))
            ).send_keys(self.email)
            
            self.driver.find_element(By.ID, "password").send_keys(self.password)
            self.driver.find_element(By.CSS_SELECTOR, "button[type='submit']").click()
            time.sleep(3)  # Wait for login to complete
            return True
        except Exception as e:
            print(f"Login failed: {str(e)}")
            return False

    def search_jobs(self, num_pages=5):
        try:
            search_url = "https://www.linkedin.com/jobs/search/?keywords=software%20engineer&location=United%20States"
            self.driver.get(search_url)
            time.sleep(3)

            for page in range(num_pages):
                print(f"Scraping page {page + 1}")
                
                # Scroll to load all jobs on the page
                self._scroll_page()
                
                # Get all job cards
                job_cards = self.driver.find_elements(By.CSS_SELECTOR, ".job-card-container")
                
                for card in job_cards:
                    try:
                        # Click on the job card to load details
                        card.click()
                        time.sleep(2)
                        
                        job_data = self._extract_job_data()
                        if job_data:
                            self.jobs_data.append(job_data)
                    
                    except Exception as e:
                        print(f"Error processing job card: {str(e)}")
                        continue

                # Click next page if available
                try:
                    next_button = self.driver.find_element(By.CSS_SELECTOR, "button[aria-label='Next']")
                    if not next_button.is_enabled():
                        break
                    next_button.click()
                    time.sleep(3)
                except:
                    break

        except Exception as e:
            print(f"Error in search_jobs: {str(e)}")

    def _scroll_page(self):
        last_height = self.driver.execute_script("return document.body.scrollHeight")
        while True:
            self.driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
            time.sleep(2)
            new_height = self.driver.execute_script("return document.body.scrollHeight")
            if new_height == last_height:
                break
            last_height = new_height

    def _extract_job_data(self):
        try:
            job_title = WebDriverWait(self.driver, 10).until(
                EC.presence_of_element_located((By.CSS_SELECTOR, ".job-details-jobs-unified-top-card__job-title"))
            ).text
            
            company = self.driver.find_element(By.CSS_SELECTOR, 
                ".job-details-jobs-unified-top-card__company-name").text
            
            location = self.driver.find_element(By.CSS_SELECTOR, 
                ".job-details-jobs-unified-top-card__bullet").text
            
            description = self.driver.find_element(By.CSS_SELECTOR, 
                ".jobs-description__content").text

            return {
                "title": job_title,
                "company": company,
                "location": location,
                "description": description,
                "scraped_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }
        except Exception as e:
            print(f"Error extracting job data: {str(e)}")
            return None

    def save_results(self):
        if not os.path.exists('data'):
            os.makedirs('data')
            
        filename = f'data/linkedin_jobs_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(self.jobs_data, f, ensure_ascii=False, indent=2)
        print(f"Results saved to {filename}")

    def close(self):
        if self.driver:
            self.driver.quit()

def main():
    # Load credentials from environment variables
    email = os.getenv('LINKEDIN_EMAIL')
    password = os.getenv('LINKEDIN_PASSWORD')
    
    if not email or not password:
        print("Please set LINKEDIN_EMAIL and LINKEDIN_PASSWORD environment variables")
        return

    scraper = LinkedInJobScraper(email, password)
    
    try:
        scraper.setup_driver()
        if scraper.login():
            scraper.search_jobs(num_pages=5)  # Scrape 5 pages of results
            scraper.save_results()
        else:
            print("Failed to login to LinkedIn")
    finally:
        scraper.close()

if __name__ == "__main__":
    main() 