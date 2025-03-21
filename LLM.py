import os
from langchain_openai import ChatOpenAI
from langchain_ollama import ChatOllama
from myAPI import *
from prompts import OLLAMA_SYSTEM_PROMPT
from LLMmodel import *

class LLMProvider:
    """Class to manage different LLM providers"""
    
    @staticmethod
    def get_llm(provider="anthropic", temperature=0.7):
        """
        Get LLM instance based on provider
        Args:
            provider (str): 'anthropic', 'openai', 'gemma', 'deepseek'
            temperature (float): Temperature for generation
        Returns:
            LLM instance
        """
        provider = provider.lower()
        
        if provider in ["anthropic", "openai"]:
            if not api_key:
                raise ValueError("API_KEY must be set in environment variables")
        
        if provider == "anthropic":
            anthropic_base_url = "https://api2.qyfxw.cn/v1"   
            model=anthropic_model
            print(f"Using anthropic model: {model}")
            print(f"Using base url: {anthropic_base_url}")
            try:
                return ChatOpenAI(
                    model=model,
                    temperature=temperature,
                    openai_api_key=api_key,
                    base_url=anthropic_base_url
                )
            except Exception as e:
                raise ValueError(f"Error initializing Anthropic model: {str(e)}")
            
        elif provider == "openai":
            openai_base_url = "https://api2.qyfxw.cn/v1" 
            model = openai_model
            print(f"Using openai model: {model}")
            print(f"Using base url: {openai_base_url}")
            try:
                # Add headers for O1-Mini specific requirements
                return ChatOpenAI(
                    model=model,
                    temperature=1.0,  # O1-Mini only supports temperature=1.0
                    openai_api_key=api_key,
                    base_url=openai_base_url
                )
            except Exception as e:
                raise ValueError(f"Error initializing OpenAI model: {str(e)}")
            
        elif provider == "gemma":
            return ChatOllama(
                model=gemma_model,
                temperature=temperature,
                base_url="http://localhost:11434",
                stop=None,
                streaming=False,
                seed=None,
                system=OLLAMA_SYSTEM_PROMPT
            )
        elif provider == "deepseek":
            return ChatOllama(
                model=deepseek_model,
                temperature=temperature,
                base_url="http://localhost:11434",
                stop=None,
                streaming=False,
                seed=None,
                system=OLLAMA_SYSTEM_PROMPT
            )
        
        else:
            raise ValueError(f"Unsupported LLM provider: {provider}")