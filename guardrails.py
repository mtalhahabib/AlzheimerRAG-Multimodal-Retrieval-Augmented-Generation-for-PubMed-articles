import requests

# ------------------------
# 1. Check for inappropriate input
# ------------------------
def is_inappropriate_input(user_input: str) -> bool:
    offensive_keywords = ["idiot", "stupid", "kill", "sex", "racist", "nude", "bomb"]
    unrelated_topics = ["bitcoin", "politics", "stock market", "religion", "celebrity"]

    user_input_lower = user_input.lower()

    for word in offensive_keywords + unrelated_topics:
        if word in user_input_lower:
            return True
    return False

# ------------------------
# 2. Preprocess user input
# ------------------------
def preprocess_input(user_input: str) -> str:
    if is_inappropriate_input(user_input):
        return "⚠️ I'm only able to assist with safe and relevant questions. Please rephrase your query."
    return None  # Input is OK

# ------------------------
# 3. Check for unsafe output from LLM
# ------------------------
def is_unsafe_output(llm_output: str) -> bool:
    unsafe_phrases = [
        "you should take this medicine",
        "kill yourself",
        "buy this stock",
        "here's legal advice",
        "this will cure your disease"
    ]
    for phrase in unsafe_phrases:
        if phrase in llm_output.lower():
            return True
    return False

# ------------------------
# 4. Postprocess model output
# ------------------------
def postprocess_output(llm_output: str) -> str:
    if is_unsafe_output(llm_output):
        return "⚠️ This content may be unsafe or misleading. Please consult a professional."
    return llm_output.strip()

# ------------------------
# 5. Main Assistant Function
# ------------------------
def guarded_virtual_assistant(query):
    # Step 1: Pre-check the input
    preprocessed = preprocess_input(query)
    if preprocessed:
        return preprocessed  # early return if unsafe input

    # Step 2: Build prompt
    prompt = f"""You are a domain expert. Answer professionally and safely.

User Query: {query}
"""

    # Step 3: Prepare payload
    payload = {
        "messages": [{"role": "user", "content": prompt}],
        "model": "meta-llama/llama-3.1-8b-instruct"
    }

    # Step 4: Make the API call
    try:
        response = requests.post(
            "https://router.huggingface.co/novita/v3/openai/chat/completions",
            headers={"Authorization": "Bearer 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"},
            json=payload
        )

        # Step 5: Parse and check model output
        if response.status_code == 200:
            llm_output = response.json()[0]["generated_text"]
            clean_output = postprocess_output(llm_output)
            return clean_output
        else:
            return f"⚠️ Error from LLM API: {response.status_code}"

    except Exception as e:
        return f"⚠️ An error occurred: {str(e)}"

# ------------------------
# Example usage
# ------------------------
if __name__ == "__main__":
    user_input = input("Ask the assistant: ")
    assistant_response = guarded_virtual_assistant(user_input)
    print("\n🧠 Assistant's Response:\n", assistant_response)
