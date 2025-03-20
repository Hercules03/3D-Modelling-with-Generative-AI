from langchain_openai import ChatOpenAI
from langchain_community.chat_models import ChatOllama
from langchain.schema.messages import HumanMessage
from myAPI import *


def test_claude():
    """Test Claude API connection"""
    testing_base_url = "https://api2.qyfxw.cn/v1"
    print("\nTesting Claude API connection...")
    
    print(f"Using base URL: {testing_base_url}")
    
    try:
        # Initialize ChatOpenAI for Claude
        chat = ChatOpenAI(
            model="claude-3-5-sonnet-20240620",
            temperature=0.7,
            openai_api_key=api_key,
            base_url=testing_base_url
        )
        
        # Simple test prompt using HumanMessage
        message = HumanMessage(content="Say 'Hello from Claude!' if you can receive this message.")
        response = chat.invoke([message])
        
        print("\nResponse from Claude:")
        print("-" * 40)
        print(response.content)
        print("-" * 40)
        
        print("✓ Claude test completed successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Claude Error: {str(e)}")
        return False

def test_openai():
    """Test Ollama (O1-Mini) connection"""
    print("\nTesting O1-Mini connection...")
    testing_base_url = "https://api2.qyfxw.cn/v1"
    
    print(f"Using base URL: {testing_base_url}")
    try:
        # Initialize ChatOllama directly
        chat = ChatOpenAI(
            model="o1-mini",
            temperature=1,
            openai_api_key=api_key,
            base_url=testing_base_url
        )
        
        # Simple test prompt using HumanMessage
        message = HumanMessage(content="Say 'Hello from O1-Mini!' if you can receive this message.")
        response = chat.invoke([message])
        
        print("\nResponse from O1-Mini:")
        print("-" * 40)
        print(response.content)
        print("-" * 40)
        
        print("✓ O1-Mini test completed successfully!")
        return True
        
    except Exception as e:
        print(f"❌ O1-Mini Error: {str(e)}")
        return False

def run_all_tests():
    """Run tests for all models"""
    print("Starting API connection tests...")
    
    claude_result = test_claude()
    ollama_result = test_openai()
    
    print("\nTest Summary:")
    print("-" * 40)
    print(f"Claude: {'✓ Success' if claude_result else '❌ Failed'}")
    print(f"O1-Mini: {'✓ Success' if ollama_result else '❌ Failed'}")
    print("-" * 40)

if __name__ == "__main__":
    run_all_tests() 