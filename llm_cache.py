"""
LLM Caching Module - Provides caching functionality for LLM calls
"""

import os
import logging
from typing import Optional, Literal
from langchain.globals import set_llm_cache  # This import is still correct
# Update these imports
from langchain_community.cache import InMemoryCache, SQLiteCache  # Updated import path

# Configure logging
logger = logging.getLogger(__name__)

class LLMCacheManager:
    """Manager for LLM caching configurations"""
    
    def __init__(self, cache_type: Literal["memory", "sqlite", "none"] = "memory", 
                 sqlite_path: Optional[str] = None):
        """
        Initialize the LLM cache manager.
        
        Args:
            cache_type: Type of cache to use ('memory', 'sqlite', or 'none')
            sqlite_path: Path to SQLite database file (only used if type is 'sqlite')
        """
        self.cache_type = cache_type
        self.sqlite_path = sqlite_path or "llm_cache.db"
        self.cache = None
        self._setup_cache()
        
    def _setup_cache(self):
        """Set up the appropriate cache based on the cache type"""
        try:
            if self.cache_type == "memory":
                logger.info("Setting up in-memory LLM cache")
                self.cache = InMemoryCache()
                set_llm_cache(self.cache)
                logger.info("In-memory LLM cache initialized")
                
            elif self.cache_type == "sqlite":
                logger.info(f"Setting up SQLite LLM cache at {self.sqlite_path}")
                self.cache = SQLiteCache(database_path=self.sqlite_path)
                set_llm_cache(self.cache)
                logger.info("SQLite LLM cache initialized")
                
            elif self.cache_type == "none":
                logger.info("LLM caching disabled")
                set_llm_cache(None)
                
            else:
                logger.warning(f"Unknown cache type: {self.cache_type}, defaulting to in-memory cache")
                self.cache = InMemoryCache()
                set_llm_cache(self.cache)
                
        except Exception as e:
            logger.error(f"Error setting up LLM cache: {e}")
            logger.info("Falling back to no cache")
            set_llm_cache(None)
            
    def clear_cache(self):
        """Clear the current cache"""
        logger.info(f"Clearing {self.cache_type} cache")
        if self.cache_type == "sqlite" and os.path.exists(self.sqlite_path):
            try:
                # For SQLite, we need to reconnect to reset the cache
                self.cache = SQLiteCache(database_path=self.sqlite_path)
                self.cache.clear()
                logger.info("SQLite cache cleared")
            except Exception as e:
                logger.error(f"Error clearing SQLite cache: {e}")
        elif self.cache_type == "memory" and self.cache:
            # For in-memory cache, we can just create a new one
            self.cache = InMemoryCache()
            set_llm_cache(self.cache)
            logger.info("In-memory cache cleared")
            
    def change_cache_type(self, new_type: Literal["memory", "sqlite", "none"], 
                         sqlite_path: Optional[str] = None):
        """
        Change the cache type.
        
        Args:
            new_type: New cache type to use
            sqlite_path: Optional new path for SQLite database
        """
        logger.info(f"Changing cache type from {self.cache_type} to {new_type}")
        self.cache_type = new_type
        if sqlite_path:
            self.sqlite_path = sqlite_path
        self._setup_cache()