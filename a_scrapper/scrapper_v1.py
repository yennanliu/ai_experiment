import cloudscraper
from bs4 import BeautifulSoup
import re


def to_base_n(n, base):
    if n == 0:
        return '0'
    alphabet = "0123456789abcdefghijklmnopqrstuvwxyz"
    res = ''
    while n > 0:
        n, rem = divmod(n, base)
        res = alphabet[rem] + res
    return res


def solve_packed_js(packed_js):
    try:
        # A simple heuristic to target the correct script
        if 'm3u8' not in packed_js:
            return []

        match = re.search(r"}\((.+)\)\)", packed_js, re.DOTALL)
        if not match:
            return []

        args_str = match.group(1)

        # A more robust way to parse the arguments
        match = re.search(r",(\d+),(\d+),'(.*?)'\.split", args_str)
        if not match:
            return []
        
        a_str, c_str, k_str = match.groups()
        a = int(a_str)
        c = int(c_str)
        k = k_str.split('|')
        
        # p is everything before this match
        p = args_str[:match.start()].strip("'")

        for i in range(c - 1, -1, -1):
            if i < len(k) and k[i]:
                key = to_base_n(i, a)
                p = re.sub(r'\b' + re.escape(key) + r'\b', k[i], p)

        urls = re.findall(r'https?://[^\']+', p)
        return [url for url in urls if url.endswith('.m3u8')]
    except Exception as e:
        print(f"Error decoding JS: {e}")
        return []

# URL to scrape
url = ''

# Create a cloudscraper instance
scraper = cloudscraper.create_scraper()

# Send GET request
response = scraper.get(url)

# Check response status
if response.status_code == 200:
    soup = BeautifulSoup(response.content, 'html.parser')

    # Find all <video> tags or <a> tags that might contain video URLs
    video_links = []
    scripts = soup.find_all('script')
    for script in scripts:
        if script.string and 'eval(function(p,a,c,k,e,d)' in script.string:
            video_links.extend(solve_packed_js(script.string))

    # Remove duplicates
    video_links = list(set(video_links))

    # Print results
    print("Found video URLs:")
    for link in video_links:
        print(link)
else:
    print(f"Failed to fetch page. Status code: {response.status_code}")
    print("Response content:")
    print(response.text)