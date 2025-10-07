import os
import time
from langchain_openai import ChatOpenAI
from google import genai
from google.genai.types import GenerateContentConfig
from together import Together


class GuardrailLLM():
    def __init__(self, model: str = "gpt-4o-mini"):
        self.model = model
        if model == "gpt-4o-mini":
            self.client = ChatOpenAI(model=model)
        elif model == "gemma-3n-e4b-it":
            self.client = genai.Client(api_key=os.environ["GOOGLE_API_KEY"])
        elif model == "OpenAI/gpt-oss-20B":
            self.client = Together() 
        pass

    def invoke(self, messages):
        print("Messages: ", messages)
        class Response:
            def __init__(self):
                self.content = None

        r = Response()
        
        if self.model == "gpt-4o-mini":
            return self.client.invoke(messages)
        elif self.model == "gemma-3n-e4b-it":
            
            time.sleep(5) #Gemma has a hard cap of 30 rpm
            response = self.client.models.generate_content(
                model="gemma-3n-e4b-it",
                contents=messages,
                config=GenerateContentConfig(
                    temperature=0,
                )
            )
            r.content = response.text
            print("Model: ", self.model)
            return r
        elif self.model == "OpenAI/gpt-oss-20B":
            response = self.client.chat.completions.create(
            model="OpenAI/gpt-oss-20B",
            temperature=0,
            messages=[
                {
                    "role": "user",
                    "content": messages.text
                }]
                )
            r.content = response.choices[0].message.content
            print("Model: ", self.model)
            return r


if __name__ == "__main__":
    import dotenv   

    print("Testing GuardrailLLM APIs...")
    dotenv.load_dotenv('.env')  # read local .env file

    # Test 1: GPT-4o-mini
    try:
        print("\n1. Testing gpt-4o-mini...")
        llm = GuardrailLLM(model="gpt-4o-mini")
        response = llm.invoke("What is 2+2?")
        print(f"Response: {response.content}")
        print("✓ gpt-4o-mini test passed")
    except Exception as e:
        print(f"✗ gpt-4o-mini test failed: {e}")
    
    # Test 2: Gemma
    try:
        print("\n2. Testing gemma-3n-e4b-it...")
        llm = GuardrailLLM(model="gemma-3n-e4b-it")
        response = llm.invoke("What is the capital of France?")
        print(f"Response: {response.content}")
        print("✓ gemma-3n-e4b-it test passed")
    except Exception as e:
        print(f"✗ gemma-3n-e4b-it test failed: {e}")
    
    # Test 3: Together API
    try:
        print("\n3. Testing OpenAI/gpt-oss-20B...")
        llm = GuardrailLLM(model="OpenAI/gpt-oss-20B")
        message = lambda: None
        message.text = "What is 1+1?"
        response = llm.invoke(message)
        print(f"Response: {response.content}")
        print("✓ OpenAI/gpt-oss-20B test passed")
    except Exception as e:
        print(f"✗ OpenAI/gpt-oss-20B test failed: {e}")
    
    print("\nAll tests completed!")

