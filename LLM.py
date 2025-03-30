"""
LLM Management Module - Consolidates LLM provider and model definitions
"""

import os
import time
import subprocess
from langchain_openai import ChatOpenAI
from langchain_ollama import ChatOllama
from typing import Optional, Literal
from dotenv import load_dotenv
import logging
import json
from pathlib import Path

# Configure logging
logger = logging.getLogger(__name__)


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


class ModelGeneratorConfig:
    """Configuration class for the 3D Model Generator"""
    def __init__(self, GenerationSettings):
        logger.info("Working on ModelGeneratorConfig Class")
        self.quit_words = ['quit', 'exit', 'bye', 'q']
        self.llm_providers = {
            "1": {"name": "anthropic", "model": ModelDefinitions.ANTHROPIC},
            "2": {"name": "openai", "model": ModelDefinitions.OPENAI},
            "3": {"name": "gemma", "model": ModelDefinitions.GEMMA},
            "4": {"name": "deepseek", "model": ModelDefinitions.DEEPSEEK}
        }
        self.settings = GenerationSettings
        self.output_dir = Path("output")
        self.output_dir.mkdir(exist_ok=True)

class OllamaManager:
    """Manager class for Ollama-related operations"""
    logger.info("Working on OllamaManager Class")
    @staticmethod
    def check_ollama(llm_provider: str) -> bool:
        """Check if Ollama is installed and running"""
        try:
            response = subprocess.run(
                ['curl', '-s', 'http://localhost:11434/api/tags'],
                capture_output=True,
                text=True
            )
            
            if response.returncode != 0:
                logger.error("Failed to connect to Ollama")
                return False
                
            model_name = {
                "gemma": ModelDefinitions.GEMMA,
                "deepseek": ModelDefinitions.DEEPSEEK
            }.get(llm_provider, ModelDefinitions.GEMMA)
            
            model_list = json.loads(response.stdout)
            if not any(model['name'] == model_name for model in model_list['models']):
                logger.warning(f"{model_name} not found. Please run: ollama pull {model_name}")
                return False
            print(f"Ollama is running and {model_name} is available")
            logger.info(f"Ollama is running and {model_name} is available")
            return True
            
        except Exception as e:
            logger.error(f"Error checking Ollama: {str(e)}")
        return False


class LLMProvider:
    """Enhanced LLM provider management"""
    logger.info("Working on LLMProvider Class")
    def __init__(self):
        logger.info("Initializing LLM Provider...")
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