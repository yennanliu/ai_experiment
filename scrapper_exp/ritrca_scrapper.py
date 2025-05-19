import requests
from bs4 import BeautifulSoup
import re

url = 'https://ri3523.org/%E5%88%86%E5%8D%80%E8%81%AF%E7%B5%A1%E8%B3%87%E8%A8%8A/#1534235114190-ffa5cc29-9b00'

headers = {
    'User-Agent': 'Mozilla/5.0'
}

response = requests.get(url, headers=headers)
response.encoding = 'utf-8'
soup = BeautifulSoup(response.text, 'html.parser')

# Dictionary to store results by group
data = {}

# Find all headings representing group titles (e.g. 第一分區)
groups = soup.find_all(['h2', 'h3', 'h4'])  # Could adjust based on actual HTML

current_group = None

for element in soup.find_all(['p', 'h2', 'h3', 'h4']):
    text = element.get_text(strip=True)

    # Identify group name (e.g., 第一分區)
    if re.match(r'^第[一二三四五六七八九十]+分區$', text):
        current_group = text
        data[current_group] = []
    
    # Extract name + email
    if current_group and ('@' in text or 'Email' in text):
        # Try to find email and name
        email_match = re.search(r'[\w\.-]+@[\w\.-]+', text)
        print('email_match = ' + str(email_match))
        if email_match:
            email = email_match.group(0)
            # Assume name is at beginning of line
            name = re.sub(r'[\w\.-]+@[\w\.-]+', '', text).replace('Email:', '').strip(' :：')
            data[current_group].append({'name': name, 'email': email})

# Print all results
for group, entries in data.items():
    print(f'\n=== {group} ===')
    for entry in entries:
        print(f"{entry['name']}: {entry['email']}")
