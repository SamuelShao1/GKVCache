import requests
import json

# Direct HTTP request to diagnose the issue
def test_raw_request(host, port):
    url = f"http://{host}:{port}/generate"
    headers = {"Content-Type": "application/json"}
    data = {
        "inputs": "Hello, how are you?",
        "parameters": {
            "max_new_tokens": 20
        }
    }
    
    print(f"Sending raw request to {url}")
    print(f"Request data: {json.dumps(data, indent=2)}")
    
    try:
        response = requests.post(url, headers=headers, json=data)
        print(f"Response status: {response.status_code}")
        
        if response.status_code == 200:
            print(f"Raw response content: {response.text}")
            try:
                json_response = response.json()
                print(f"Parsed JSON response: {json.dumps(json_response, indent=2)}")
                return json_response
            except json.JSONDecodeError as e:
                print(f"Error parsing JSON response: {e}")
                return None
        else:
            print(f"Error response: {response.text}")
            return None
    except Exception as e:
        print(f"Request error: {e}")
        return None

# Check server health
def check_health(host, port):
    url = f"http://{host}:{port}/health"
    try:
        response = requests.get(url)
        print(f"Health check status: {response.status_code}")
        print(f"Health response: {response.text}")
        return response.status_code == 200
    except Exception as e:
        print(f"Health check error: {e}")
        return False

# Check server info
def check_info(host, port):
    url = f"http://{host}:{port}/info"
    try:
        response = requests.get(url)
        print(f"Info check status: {response.status_code}")
        if response.status_code == 200:
            print(f"Server info: {json.dumps(response.json(), indent=2)}")
        else:
            print(f"Info response: {response.text}")
        return response.status_code == 200
    except Exception as e:
        print(f"Info check error: {e}")
        return False

# Main function
def diagnostic():
    host = "104.171.202.139"
    port = 8080
    
    print(f"Diagnosing TGI server at {host}:{port}")
    
    # Check server health
    is_healthy = check_health(host, port)
    if not is_healthy:
        print("Server health check failed. The server might not be running correctly.")
    
    # Check server info
    check_info(host, port)
    
    # Try a raw generate request
    response = test_raw_request(host, port)
    
    # If we got a response, analyze it
    if response:
        if "generated_text" in response:
            if response["generated_text"] == "":
                print("The server returned an empty response. This could be due to:")
                print("1. The model is not generating text properly")
                print("2. There might be an issue with the model loading")
                print("3. The prompt might need adjustment")
            else:
                print(f"Success! Full generated text: '{response['generated_text']}'")
        else:
            print("The response doesn't contain 'generated_text'. Check the response format.")
    
    print("\nDiagnostics complete.")

if __name__ == "__main__":
    diagnostic()