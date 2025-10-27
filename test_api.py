import requests
import json

API_URL = "http://127.0.0.1:5001/predict"
IMAGE_PATH = "organized_dataset/bcc/ISIC_0024332.jpg" # Change this to your test image path

def test_prediction_endpoint():
    print(f"Testing POST request to {API_URL} with image: {IMAGE_PATH}")
    
    try:
        # Open the image file
        with open(IMAGE_PATH, 'rb') as f:
            # Prepare the files dictionary for the requests library
            files = {'file': (IMAGE_PATH, f)}
            
            # Send the POST request
            response = requests.post(API_URL, files=files)
            
        # Check the status code
        response.raise_for_status() # Raises an HTTPError for bad responses (4xx or 5xx)
        
        # Parse the JSON response
        data = response.json()
        
        # Print key results
        print("\n--- API Test Successful ---")
        print(f"Status Code: {response.status_code}")
        print(f"Primary Diagnosis: {data['diagnosis']['disease']}")
        print(f"Confidence: {data['diagnosis']['confidence']:.2f}%")
        print(f"Cancer Status: {data['diagnosis']['cancerStatus']}")
        print("----------------------------")
        
        # Optional: Print the full explanation structure
        # print(json.dumps(data, indent=4))
        
    except requests.exceptions.RequestException as e:
        print(f"\n--- API Test FAILED ---")
        print(f"Error connecting to API or bad response: {e}")
        if 'response' in locals():
            print(f"Response Content (Error): {response.text}")
        
    except FileNotFoundError:
        print(f"\n--- API Test FAILED ---")
        print(f"Error: The image file '{IMAGE_PATH}' was not found.")

if __name__ == "__main__":
    test_prediction_endpoint()