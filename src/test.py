import requests

def fetch_local_api(url):
    try:
        response = requests.get(url)
        if response.status_code == 200:
            # If the request was successful (status code 200),
            # you can access the response data using response.json()
            data = response.json()
            return data
        else:
            print("Error: Unable to fetch data from API. Status code:", response.status_code)
            return None
    except requests.exceptions.RequestException as e:
        print("Error: Request failed:", e)
        return None

# Example usage
url = "http://localhost:3000/landmarks"  # Replace this with the URL of your local API
data = fetch_local_api(url)
if data:
    print("Data fetched successfully:", data)
