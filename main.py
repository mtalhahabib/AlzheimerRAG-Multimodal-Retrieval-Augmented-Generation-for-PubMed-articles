import requests
import time

API_URL = "https://api-inference.huggingface.co/models/meta-llama/Meta-Llama-3-8B-Instruct"
HEADERS = {"Authorization": "Bearer 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"}

def query(payload, retries=3, wait_time=10):
    """Send request to Hugging Face Inference API with retry logic."""
    for attempt in range(retries):
        response = requests.post(API_URL, headers=HEADERS, json=payload)
        
        if response.status_code == 200:
            return response.json()
        elif response.status_code == 503:  # Model too busy
            print(f"⚠️ Model is busy. Retrying in {wait_time} seconds... ({attempt+1}/{retries})")
            time.sleep(wait_time)
        else:
            print(f"❌ Error: {response.status_code}, {response.text}")
            break
    
    return {"error": "Failed after multiple retries"}

data = query({
    "inputs": "Explain Alzheimer's disease in simple terms.",
    "parameters": {
        "max_new_tokens": 100,  # Reduce output size for faster response
        "temperature": 0.3,  # Reduce randomness
        "top_p": 0.9,  # Nucleus sampling
        "do_sample": False  # Disable sampling for consistent output
    }
})

print(data)
