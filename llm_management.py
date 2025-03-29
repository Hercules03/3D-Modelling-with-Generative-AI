"""
LLM Management Module - Consolidates LLM provider and model definitions
"""

import os
import time
from langchain_openai import ChatOpenAI
from langchain_ollama import ChatOllama
from typing import Optional, Literal
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Model Definitions
class ModelDefinitions:
    """Centralized model definitions"""
    ANTHROPIC = "claude-3-7-sonnet-20250219"
    OPENAI = "o3-mini"
    GEMMA = "gemma3:4b-it-q8_0"
    DEEPSEEK = "deepseek-r1:7b"
    
    # Special purpose models
    VALIDATOR = "gemma3:1b"
    KEYWORD_EXTRACTOR = "keyword-extractor:latest"
    
    @classmethod
    def get_model(cls, provider: str, purpose: Optional[str] = None) -> str:
        """Get the appropriate model for a provider and purpose"""
        if purpose:
            if purpose == "validation":
                return cls.VALIDATOR
            elif purpose == "keyword_extraction":
                return cls.KEYWORD_EXTRACTOR
        
        provider_map = {
            "anthropic": cls.ANTHROPIC,
            "openai": cls.OPENAI,
            "gemma": cls.GEMMA,
            "deepseek": cls.DEEPSEEK
        }
        return provider_map.get(provider.lower())

class LLMProvider:
    """Enhanced LLM provider management"""
    
    def __init__(self):
        self.models = ModelDefinitions()
    
    @staticmethod
    def get_llm(
        provider: str = "anthropic",
        temperature: float = 0.7,
        max_retries: int = 3,
        model: Optional[str] = None,
        purpose: Optional[str] = None
    ):
        """
        Get LLM instance based on provider and purpose
        
        Args:
            provider: 'anthropic', 'openai', 'gemma', 'deepseek'
            temperature: Temperature for generation
            max_retries: Maximum number of retries on failure
            model: Optional specific model to use (overrides default)
            purpose: Optional specific purpose (e.g., 'validation', 'keyword_extraction')
        
        Returns:
            LLM instance
        """
        provider = provider.lower()
        
        # Get API key for cloud providers
        if provider in ["anthropic", "openai"]:
            api_key = os.getenv("API_KEY")
            if not api_key:
                raise ValueError(
                    "API_KEY environment variable is not set. Please add your API key to the .env file:\n"
                    "1. Create or edit the .env file in the project root\n"
                    "2. Add the line: API_KEY=your_api_key_here\n"
                    "3. Replace 'your_api_key_here' with your actual API key"
                )
        
        # Get appropriate model
        if not model:
            model = ModelDefinitions.get_model(provider, purpose)
        
        retry_count = 0
        while retry_count < max_retries:
            try:
                if provider == "anthropic":
                    anthropic_base_url = os.getenv("ANTHROPIC_BASE_URL", "https://api2.qyfxw.cn/v1")
                    print(f"Using anthropic model: {model}")
                    print(f"Using base url: {anthropic_base_url}")
                    return ChatOpenAI(
                        model=model,
                        temperature=temperature,
                        openai_api_key=api_key,
                        base_url=anthropic_base_url,
                        streaming=True 
                    )
                    
                elif provider == "openai":
                    openai_base_url = os.getenv("OPENAI_BASE_URL", "https://api2.qyfxw.cn/v1")
                    print(f"Using openai model: {model}")
                    print(f"Using base url: {openai_base_url}")
                    return ChatOpenAI(
                        model=model,
                        temperature=1.0,  # O1-Mini only supports temperature=1.0
                        openai_api_key=api_key,
                        base_url=openai_base_url,
                        streaming=True
                    )
                    
                elif provider == "gemma":
                    print(f"Using Gemma model: {model}")
                    return ChatOllama(
                        model=model,
                        temperature=temperature,
                        base_url="http://localhost:11434",
                        stop=None,
                        seed=None,
                        system=None,  # System prompt should be passed separately
                        streaming=True
                    )
                    
                elif provider == "deepseek":
                    print(f"Using Deepseek model: {model}")
                    return ChatOllama(
                        model=model,
                        temperature=temperature,
                        base_url="http://localhost:11434",
                        stop=None,
                        seed=None,
                        system=None,  # System prompt should be passed separately
                        streaming=True
                    )
                
                else:
                    raise ValueError(f"Unsupported LLM provider: {provider}")
                    
            except Exception as e:
                retry_count += 1
                if retry_count < max_retries:
                    print(f"\nError connecting to {provider} (attempt {retry_count}/{max_retries}): {str(e)}")
                    print("Retrying in 5 seconds...")
                    time.sleep(5)
                    continue
                else:
                    print(f"\nFailed to connect to {provider} after {max_retries} attempts.")
                    print("Falling back to local Ollama model (Gemma)...")
                    return ChatOllama(
                        model=ModelDefinitions.GEMMA,
                        temperature=temperature,
                        base_url="http://localhost:11434",
                        stop=None,
                        seed=None,
                        system=None,
                        streaming=True
                    ) 