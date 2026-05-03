import requests
import json
import time
import random
from bs4 import BeautifulSoup

def scrape_amazon_ps5():
    base_url = "https://www.amazon.com"
    search_url = "https://www.amazon.com/s?k=ps5"
    
    headers = {
        "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/117.0.0.0 Safari/537.36",
        "Accept-Language": "en-US,en;q=0.5",
    }

    products = []
    page_url = search_url
    
    for i in range(2): # Try up to 2 pages
        if len(products) >= 30: break
        print(f"Fetching: {page_url}")
        try:
            response = requests.get(page_url, headers=headers, timeout=15)
            if response.status_code != 200:
                print(f"Failed: {response.status_code}")
                break
            
            if "To discuss automated access" in response.text or "bm-verify" in response.text:
                print("Blocked.")
                break

            soup = BeautifulSoup(response.text, 'html.parser')
            items = soup.select('div[data-asin]')
            
            for item in items:
                if len(products) >= 30: break
                asin = item.get('data-asin')
                if not asin: continue
                
                title_tag = item.select_one('h2 span')
                title = title_tag.text.strip() if title_tag else "N/A"
                
                price_tag = item.select_one('span.a-price span.a-offscreen')
                price = price_tag.text.strip() if price_tag else "N/A"
                
                rating_tag = item.select_one('span.a-icon-alt')
                rating = rating_tag.text.strip() if rating_tag else "N/A"
                
                review_tag = item.select_one('a[href*="#customerReviews"]')
                review_count = "0"
                if review_tag:
                    aria = review_tag.get('aria-label', '')
                    if 'ratings' in aria:
                        review_count = aria.split()[0]
                    else:
                        review_count = review_tag.text.strip().replace('(', '').replace(')', '')

                link_tag = item.select_one('h2 a')
                product_url = base_url + link_tag['href'] if link_tag else "N/A"
                
                is_sponsored = "Sponsored" in item.get_text()
                
                products.append({
                    "title": title,
                    "price": price,
                    "rating": rating,
                    "review_count": review_count,
                    "product_url": product_url,
                    "is_sponsored": is_sponsored
                })
            
            print(f"Count: {len(products)}")
            
            if len(products) < 30:
                next_pg = soup.select_one('a.s-pagination-next')
                if next_pg:
                    page_url = base_url + next_pg['href']
                    time.sleep(random.uniform(5, 10))
                else:
                    break
        except Exception as e:
            print(f"Error: {e}")
            break

    if products:
        output_path = "/Users/jliu/ai_experiment/agent_team/app2/output_gemini/amazon_ps5_products.json"
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(products, f, indent=4, ensure_ascii=False)
        print(f"Saved {len(products)} items.")
    else:
        print("No products to save.")

if __name__ == "__main__":
    scrape_amazon_ps5()
