# from huggingface_hub import InferenceClient

# # Replace with your HF token
# client = InferenceClient(token="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")

# # Choose a hosted model — e.g., Falcon, Mistral, GPT2, Llama, etc.
# response = client.text_generation(
#     prompt="Explain how quantum computers work.",
#     model="HuggingFaceH4/zephyr-7b-beta",  # you can change to any hosted model
#     max_new_tokens=100,
#     temperature=0.7
# )

# # print(response)
# import requests

# API_URL = "https://router.huggingface.co/novita/v3/openai/chat/completions"
# headers = {"Authorization": "Bearer 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"}

# payload = {
#     "inputs": "Can you explain general relativity?",
#     "parameters": {
#         "max_new_tokens": 50,
#         "temperature": 0.7
#     }
# }

# response = requests.post(API_URL, headers=headers, json=payload)

# # Debug output
# print("Status code:", response.status_code)
# print("Response text:", response.text)

# # Safe JSON parsing
# try:
#     print(response.json())
# except Exception as e:
#     print("Failed to parse JSON:", str(e))
import requests

API_URL = "https://router.huggingface.co/novita/v3/openai/chat/completions"
headers = {
    "Authorization": "Bearer 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",
}

def query(payload):
    response = requests.post(API_URL, headers=headers, json=payload)
    return response.json()

response = query({
    "messages": [
        {
            "role": "user",
            "content": "What is the capital of France?"
        }
    ],
    "model": "meta-llama/llama-3.1-8b-instruct"
})

print(response["choices"][0]["message"]['content'])