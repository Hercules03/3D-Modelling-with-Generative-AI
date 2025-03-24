import os
from langchain_anthropic import ChatAnthropic
from langchain_openai import ChatOpenAI
from langchain_ollama import ChatOllama
from myAPI import *
from prompts import OLLAMA_SYSTEM_PROMPT
from LLMmodel import *
import time

class LLMProvider:
    """Class to manage different LLM providers"""
    
    @staticmethod
    def get_llm(provider="anthropic", temperature=0.7, max_retries=3, model=None):
        """
        Get LLM instance based on provider
        Args:
            provider (str): 'anthropic', 'openai', 'gemma', 'deepseek'
            temperature (float): Temperature for generation
            max_retries (int): Maximum number of retries on failure
            model (str): Optional specific model to use (overrides default)
        Returns:
            LLM instance
        """
        provider = provider.lower()
        
        if provider in ["anthropic", "openai"]:
            if not api_key:
                raise ValueError("API_KEY must be set in environment variables")
        
        retry_count = 0
        while retry_count < max_retries:
            try:
                if provider == "anthropic":
                    anthropic_base_url = "https://api2.qyfxw.cn/v1"   
                    model = model or anthropic_model
                    print(f"Using anthropic model: {model}")
                    print(f"Using base url: {anthropic_base_url}")
                    try:
                        return ChatOpenAI(
                            model=model,
                            temperature=temperature,
                            openai_api_key=api_key,
                            base_url=anthropic_base_url,
                            streaming=True 
                        )
                    except Exception as e:
                        raise ValueError(f"Error initializing Anthropic model: {str(e)}")
                    
                elif provider == "openai":
                    openai_base_url = "https://api2.qyfxw.cn/v1" 
                    model = model or openai_model
                    print(f"Using openai model: {model}")
                    print(f"Using base url: {openai_base_url}")
                    try:
                        return ChatOpenAI(
                            model=model,
                            temperature=1.0,  # O1-Mini only supports temperature=1.0
                            openai_api_key=api_key,
                            base_url=openai_base_url,
                            streaming=True
                        )
                    except Exception as e:
                        raise ValueError(f"Error initializing OpenAI model: {str(e)}")
                    
                elif provider == "gemma":
                    # Use provided model or fall back to default
                    use_model = model or gemma_model
                    print(f"Using Gemma model: {use_model}")
                    return ChatOllama(
                        model=use_model,
                        temperature=temperature,
                        base_url="http://localhost:11434",
                        stop=None,
                        seed=None,
                        system=OLLAMA_SYSTEM_PROMPT,
                        streaming=True
                    )
                    
                elif provider == "deepseek":
                    model = model or deepseek_model
                    print(f"Using Deepseek model: {model}")
                    return ChatOllama(
                        model=model,
                        temperature=temperature,
                        base_url="http://localhost:11434",
                        stop=None,
                        seed=None,
                        system=OLLAMA_SYSTEM_PROMPT,
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
                        model=model or gemma_model,
                        temperature=temperature,
                        base_url="http://localhost:11434",
                        stop=None,
                        seed=None,
                        system=OLLAMA_SYSTEM_PROMPT,
                        streaming=True
                    )