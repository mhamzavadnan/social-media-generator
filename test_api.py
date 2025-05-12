import os
from dotenv import load_dotenv
import openai

def test_api_key():
    # Load environment variables
    load_dotenv('config.env')
    
    # Get API key
    api_key = os.getenv('OPENAI_API_KEY')
    
    if not api_key:
        print("❌ Error: No API key found in config.env")
        return
    
    # Set up OpenAI client
    openai.api_key = api_key
    
    try:
        # Try a simple API call
        response = openai.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": "Hello!"}],
            max_tokens=10
        )
        print("✅ API key is working correctly!")
        print("Response:", response.choices[0].message.content)
        
    except Exception as e:
        print("❌ Error testing API key:", str(e))

if __name__ == "__main__":
    test_api_key() 