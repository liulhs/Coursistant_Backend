import requests

# URL of the local API endpoint
url = "https://piazza.e-ta.net/users/login"

# Payload for the POST request
payload = {
    "email": "haosongl@usc.edu",
    "password": "Lhs2016w@"
}

# Headers for the request
headers = {
    "Content-Type": "application/json"
}

# Sending the POST request
response = requests.post(url, json=payload, headers=headers)

# Print the response
print("Status Code:", response.status_code)
print("Response Body:", response.json())

get_url = "https://piazza.e-ta.net/users/haosongl@usc.edu/courses/llqyd5tpdcq34o/posts/all"
next_response = requests.get(get_url)

print("Status Code:", next_response.status_code)
print("Response Body:", next_response)
