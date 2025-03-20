import os
from langchain_openai import ChatOpenAI
from langchain_ollama import ChatOllama
from myAPI import *
from prompts import OLLAMA_SYSTEM_PROMPT

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
            if not api_key or not base_url:
                raise ValueError("API_KEY and BASE_URL must be set in environment variables")
        
        if provider == "anthropic":
            model = "claude-3-5-sonnet-20240620"
            # model="claude-3-7-sonnet-20250219"
            print(f"Using anthropic model: {model}")
            print(f"Using base url: {base_url}")
            try:
                return ChatOpenAI(
                    model=model,
                    temperature=temperature,
                    openai_api_key=api_key,
                    base_url=base_url
                )
            except Exception as e:
                raise ValueError(f"Error initializing Anthropic model: {str(e)}")
            
        elif provider == "openai":
            model = "o1-mini"
            print(f"Using openai model: {model}")
            print(f"Using base url: {base_url}")
            try:
                # Add headers for O1-Mini specific requirements
                return ChatOpenAI(
                    model=model,
                    temperature=1.0,  # O1-Mini only supports temperature=1.0
                    openai_api_key=api_key,
                    base_url=base_url
                )
            except Exception as e:
                raise ValueError(f"Error initializing OpenAI model: {str(e)}")
            
        elif provider == "gemma":
            return ChatOllama(
                model="gemma3:4b-it-q8_0",
                temperature=temperature,
                base_url="http://localhost:11434",
                stop=None,
                streaming=False,
                seed=None,
                system=OLLAMA_SYSTEM_PROMPT
            )
        elif provider == "deepseek":
            return ChatOllama(
                model="deepseek-r1:7b",
                temperature=temperature,
                base_url="http://localhost:11434",
                stop=None,
                streaming=False,
                seed=None,
                system=OLLAMA_SYSTEM_PROMPT
            )
        
        else:
            raise ValueError(f"Unsupported LLM provider: {provider}")