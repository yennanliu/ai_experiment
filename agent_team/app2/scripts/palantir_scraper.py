import requests
from bs4 import BeautifulSoup
import json
import re
import os
import time

def clean_text(text):
    # Remove citation markers like [1], [12], etc.
    text = re.sub(r'\[\d+\]', '', text)
    # Remove extra whitespace
    text = ' '.join(text.split())
    return text

def scrape_palantir_wikipedia():
    url = "https://en.wikipedia.org/wiki/Palantir_Technologies"
    headers = {
        'User-Agent': 'PalantirScraper/1.0 (Educational Scraping Project; contact: agent@example.com)'
    }

    print(f"Fetching {url}...")
    try:
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
    except Exception as e:
        print(f"Error fetching page: {e}")
        return

    soup = BeautifulSoup(response.text, 'html.parser')

    data = {
        'infobox': {},
        'history_summary': "",
        'products_summary': ""
    }

    # 1. Infobox Extraction
    infobox = soup.find('table', class_='infobox')
    if infobox:
        rows = infobox.find_all('tr')
        for row in rows:
            th = row.find('th')
            td = row.find('td')
            if th and td:
                key = th.get_text(strip=True)
                value = clean_text(td.get_text(separator=' ', strip=True))
                data['infobox'][key] = value

    # 2. History Summary
    history_heading_div = None
    for h2 in soup.find_all('h2'):
        if 'History' in h2.get_text():
            if h2.parent and h2.parent.name == 'div' and 'mw-heading' in h2.parent.get('class', []):
                history_heading_div = h2.parent
            else:
                history_heading_div = h2
            break
    
    if history_heading_div:
        paragraphs = []
        for sibling in history_heading_div.find_next_siblings():
            if sibling.name == 'h2' or (sibling.name == 'div' and 'mw-heading2' in sibling.get('class', [])):
                break
            if sibling.name == 'p':
                txt = clean_text(sibling.get_text())
                if txt:
                    paragraphs.append(txt)
            if len(paragraphs) >= 3:
                break
        data['history_summary'] = "\n\n".join(paragraphs)

    # 3. Products Summary
    products_heading_div = None
    for h2 in soup.find_all('h2'):
        if 'Products' in h2.get_text():
            if h2.parent and h2.parent.name == 'div' and 'mw-heading' in h2.parent.get('class', []):
                products_heading_div = h2.parent
            else:
                products_heading_div = h2
            break

    if products_heading_div:
        target_products = ['Gotham', 'Foundry', 'Apollo']
        found_data = {p: [] for p in target_products}
        intro_texts = []
        
        current_product = None
        for sibling in products_heading_div.find_next_siblings():
            if sibling.name == 'h2' or (sibling.name == 'div' and 'mw-heading2' in sibling.get('class', [])):
                break
            
            is_h3 = False
            h3_text = ""
            if sibling.name == 'h3':
                is_h3 = True
                h3_text = sibling.get_text().strip()
            elif sibling.name == 'div' and 'mw-heading3' in sibling.get('class', []):
                is_h3 = True
                h3_inner = sibling.find('h3')
                h3_text = h3_inner.get_text().strip() if h3_inner else sibling.get_text().strip()

            if is_h3:
                current_product = None
                for p in target_products:
                    if p.lower() in h3_text.lower():
                        current_product = p
                        break
                continue

            if sibling.name == 'p':
                txt = clean_text(sibling.get_text())
                if not txt: continue
                
                if current_product:
                    found_data[current_product].append(txt)
                else:
                    # Check if paragraph mentions any target product
                    found_any = False
                    for p in target_products:
                        if p in txt:
                            if txt not in found_data[p]:
                                found_data[p].append(txt)
                            found_any = True
                    if not found_any:
                        intro_texts.append(txt)

        summary_parts = []
        if intro_texts:
            summary_parts.append(" ".join(intro_texts))
        
        for p in target_products:
            if found_data[p]:
                summary_parts.append(f"Palantir {p}: {' '.join(found_data[p])}")

        data['products_summary'] = "\n\n".join(summary_parts)

    # Save to file
    output_dir = "/Users/jliu/ai_experiment/agent_team/app2/output_gemini"
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "palantir_wikipedia.json")

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=4)

    print(f"Data saved to {output_path}")

if __name__ == "__main__":
    scrape_palantir_wikipedia()
    time.sleep(1)
