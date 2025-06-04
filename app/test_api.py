import requests
import os
from pathlib import Path

def test_api():
    # Base URL
    base_url = "http://localhost:8000"
    
    # Test root endpoint
    print("\nTesting root endpoint...")
    response = requests.get(f"{base_url}/")
    print(f"Status: {response.status_code}")
    print(f"Response: {response.json()}")
    
    # Test extraction endpoint with a sample PDF
    print("\nTesting extraction endpoint...")
    
    # Create a test PDF file
    test_pdf_path = "test.pdf"
    with open(test_pdf_path, "w") as f:
        f.write("This is a test PDF file")
    
    try:
        # Prepare the files for upload
        files = [
            ('files', ('test.pdf', open(test_pdf_path, 'rb'), 'application/pdf'))
        ]
        
        # Make the request
        response = requests.post(
            f"{base_url}/extract",
            files=files,
            data={'part_number': 'TEST123'}
        )
        
        print(f"Status: {response.status_code}")
        print(f"Response: {response.json()}")
        
    except Exception as e:
        print(f"Error: {str(e)}")
    finally:
        # Clean up test file
        if os.path.exists(test_pdf_path):
            os.remove(test_pdf_path)

if __name__ == "__main__":
    test_api() 