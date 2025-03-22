import os
from dotenv import load_dotenv
from openai import OpenAI

def test_api_connection():
    # Load environment variables from .env file
    if not load_dotenv():
        print("Error: Could not load .env file. Make sure it exists in the current directory.")
        return

    # Get the API key and base URL from environment variables
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        print("Error: OPENROUTER_API_KEY not found in .env file.")
        return

    base_url = os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
    
    # Initialize the OpenAI client with OpenRouter
    client = OpenAI(
        base_url=base_url,
        api_key=api_key,
    )

    # Test a simple API call
    try:
        print("Testing API connection with GPT-4o-mini...")
        response = client.chat.completions.create(
            model="openai/gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "Hello! This is a test message to verify my API connection."}
            ],
            max_tokens=100
        )
        
        # Print the response
        print("\nAPI Connection Successful! ✅")
        print(f"Model: {response.model}")
        print(f"Response: {response.choices[0].message.content}")
        
    except Exception as e:
        print(f"\nAPI Connection Failed! ❌")
        print(f"Error: {str(e)}")

if __name__ == "__main__":
    test_api_connection()