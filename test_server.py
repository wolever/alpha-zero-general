import requests
import json
import numpy as np

# Test board
board_arr = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, -5, 0, 0, 0, -1, 0, 0, 0, -5,
       15, 0, 0, 0, 0, 0, 0, 0, -1, 0, 0, -1, 0, 0, 0, 0, 0,
        0, 0, 0, 0, -2, 0, 0, 0, -2, -1, 0, 0])

# Convert the board to a list for JSON serialization
board_list = board_arr.tolist()

# API URL
url = "http://localhost:8189/get-moves"

# Make the request
try:
    response = requests.post(
        url,
        json={"board": board_list},
        headers={"Content-Type": "application/json"}
    )

    # Check if request was successful
    if response.status_code == 200:
        # Print the response
        data = response.json()
        print("Moves and probabilities:")
        for move in data["moves"]:
            print(f"Type: {move['type']}")
            print(f"Source Index: {move['src_idx']}")
            print(f"Destination Index: {move['dst_idx']}")
            print(f"Count: {move['count']}")
            print(f"Weight: {move['weight']}")
            print("---")
    else:
        print(f"Error: {response.status_code}")
        print(response.text)

except Exception as e:
    print(f"Request failed: {e}")

# Also test the health endpoint
try:
    health_response = requests.get("http://localhost:8189/health")
    print(f"Health check status: {health_response.status_code}")
    print(health_response.json())
except Exception as e:
    print(f"Health check failed: {e}")
