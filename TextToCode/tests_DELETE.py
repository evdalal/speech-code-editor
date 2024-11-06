import requests
import json

def send_test_request(url: str, payload: dict) -> None:
    try:
        # Send a POST request to the server
        response = requests.post(url, json=payload)
        
        # Print the response from the server
        print(f"Status Code: {response.status_code}")
        print(f"Response: {response.json()}")
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    # Define the URL of the server endpoint
    url = "http://127.0.0.1:5000"

    # Define the JSON payload to send
    example_prompt = 'Write a function which calculates the nth Fibonnaci value.'
    payload = {'prompt': example_prompt, 'userid': 1, 'conversationid': 1}


    # Send the test request
    send_test_request(url, payload)
