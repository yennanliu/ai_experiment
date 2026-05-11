import requests

API_KEY = 'YOUR_GOOGLE_MAPS_API_KEY'

# Place ID for "Unitech Electronics Co., Ltd."
# You can obtain this from the Maps URL or search API.
PLACE_ID = 'ChIJH9kDfAilQjQRTp5msKbs1OE'

def get_place_details(place_id, api_key):
    url = f'https://maps.googleapis.com/maps/api/place/details/json'
    params = {
        'place_id': place_id,
        'fields': 'name,formatted_address,website',
        'key': api_key
    }

    response = requests.get(url, params=params)
    data = response.json()

    if data.get('status') != 'OK':
        print(f"Error: {data.get('status')}")
        return None

    result = data.get('result', {})
    return {
        'name': result.get('name', 'null'),
        'website': result.get('website', 'null'),
        'address': result.get('formatted_address', 'null')
    }

# Run the function
details = get_place_details(PLACE_ID, API_KEY)

# Output the result
print(details)
