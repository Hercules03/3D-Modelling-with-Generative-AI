"""
User Session Management Module

Provides tracking of user sessions and their generation history.
"""

import os
import json
import uuid
import logging
from datetime import datetime
from typing import Dict, List, Optional, Any, Union

# Configure logging
logger = logging.getLogger(__name__)

class UserSession:
    """
    Class to track and manage user sessions and their generation history.
    
    Attributes:
        user_id (str): Unique identifier for the user
        history (List[Dict]): List of generation history entries
        session_id (str): Unique identifier for this session
        created_at (str): ISO format timestamp when session was created
        last_active (str): ISO format timestamp when session was last active
    """
    
    def __init__(self, user_id: str = None):
        """
        Initialize a new user session.
        
        Args:
            user_id (str, optional): User identifier. If None, a random UUID will be generated
        """
        self.user_id = user_id if user_id else str(uuid.uuid4())
        self.history = []
        self.session_id = str(uuid.uuid4())
        self.created_at = datetime.now().isoformat()
        self.last_active = self.created_at
        logger.info(f"Created new session {self.session_id} for user {self.user_id}")
        
    def add_generation(self, description: str, code: str, metadata: Dict[str, Any] = None) -> Dict:
        """
        Add a new generation entry to the user's history.
        
        Args:
            description (str): The description/prompt used for generation
            code (str): The generated OpenSCAD code
            metadata (Dict, optional): Additional metadata about the generation
            
        Returns:
            Dict: The created history entry
        """
        # Update last active timestamp
        self.last_active = datetime.now().isoformat()
        
        # Create history entry
        entry = {
            "id": str(uuid.uuid4()),
            "timestamp": self.last_active,
            "description": description,
            "code": code,
            "metadata": metadata or {}
        }
        
        # Add to history
        self.history.append(entry)
        logger.info(f"Added generation {entry['id']} to session {self.session_id}")
        
        return entry
    
    def get_history(self, limit: int = None, offset: int = 0) -> List[Dict]:
        """
        Get the generation history for this user session.
        
        Args:
            limit (int, optional): Maximum number of entries to return
            offset (int, optional): Number of entries to skip
            
        Returns:
            List[Dict]: List of history entries, newest first
        """
        # Sort by timestamp, newest first
        sorted_history = sorted(
            self.history, 
            key=lambda x: x["timestamp"], 
            reverse=True
        )
        
        # Apply offset and limit
        if offset > 0:
            sorted_history = sorted_history[offset:]
        if limit is not None:
            sorted_history = sorted_history[:limit]
            
        return sorted_history
    
    def get_generation(self, generation_id: str) -> Optional[Dict]:
        """
        Retrieve a specific generation by ID.
        
        Args:
            generation_id (str): ID of the generation to retrieve
            
        Returns:
            Optional[Dict]: The generation entry, or None if not found
        """
        for entry in self.history:
            if entry["id"] == generation_id:
                return entry
        return None
    
    def to_dict(self) -> Dict:
        """
        Convert the session to a dictionary for serialization.
        
        Returns:
            Dict: Session data as a dictionary
        """
        return {
            "user_id": self.user_id,
            "session_id": self.session_id,
            "created_at": self.created_at,
            "last_active": self.last_active,
            "history": self.history
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'UserSession':
        """
        Create a UserSession from a dictionary.
        
        Args:
            data (Dict): Dictionary containing session data
            
        Returns:
            UserSession: New session object with restored data
        """
        session = cls(user_id=data["user_id"])
        session.session_id = data["session_id"]
        session.created_at = data["created_at"]
        session.last_active = data["last_active"]
        session.history = data["history"]
        return session


class SessionManager:
    """
    Manages multiple user sessions, including persistence to disk.
    
    Attributes:
        sessions (Dict[str, UserSession]): Dictionary of active sessions by session_id
        session_dir (str): Directory where session data is stored
    """
    
    def __init__(self, session_dir: str = "user_sessions"):
        """
        Initialize the session manager.
        
        Args:
            session_dir (str, optional): Directory to store session data
        """
        self.sessions = {}
        self.session_dir = session_dir
        
        # Create session directory if it doesn't exist
        os.makedirs(self.session_dir, exist_ok=True)
        logger.info(f"Session manager initialized with directory: {self.session_dir}")
        
        # Load existing sessions
        self._load_sessions()
        
    def create_session(self, user_id: str = None) -> UserSession:
        """
        Create a new user session.
        
        Args:
            user_id (str, optional): User identifier
            
        Returns:
            UserSession: The newly created session
        """
        session = UserSession(user_id=user_id)
        self.sessions[session.session_id] = session
        self._save_session(session)
        return session
    
    def get_session(self, session_id: str) -> Optional[UserSession]:
        """
        Get a user session by ID.
        
        Args:
            session_id (str): Session identifier
            
        Returns:
            Optional[UserSession]: The session, or None if not found
        """
        return self.sessions.get(session_id)
    
    def get_user_sessions(self, user_id: str) -> List[UserSession]:
        """
        Get all sessions for a specific user.
        
        Args:
            user_id (str): User identifier
            
        Returns:
            List[UserSession]: List of sessions for the user
        """
        return [s for s in self.sessions.values() if s.user_id == user_id]
    
    def save_session(self, session: UserSession) -> None:
        """
        Save a session to persistent storage.
        
        Args:
            session (UserSession): Session to save
        """
        self._save_session(session)
        
    def add_generation(self, session_id: str, description: str, code: str, 
                      metadata: Dict[str, Any] = None) -> Optional[Dict]:
        """
        Add a generation to a session.
        
        Args:
            session_id (str): Session identifier
            description (str): Generation description/prompt
            code (str): Generated OpenSCAD code
            metadata (Dict, optional): Additional metadata
            
        Returns:
            Optional[Dict]: The created history entry, or None if session not found
        """
        session = self.get_session(session_id)
        if not session:
            logger.warning(f"Session {session_id} not found")
            return None
            
        # Add the generation
        entry = session.add_generation(description, code, metadata)
        
        # Save the updated session
        self._save_session(session)
        
        return entry
        
    def _get_session_path(self, session_id: str) -> str:
        """Get the file path for a session."""
        return os.path.join(self.session_dir, f"{session_id}.json")
        
    def _save_session(self, session: UserSession) -> None:
        """Save a session to a JSON file."""
        path = self._get_session_path(session.session_id)
        try:
            with open(path, 'w') as f:
                json.dump(session.to_dict(), f, indent=2)
            logger.debug(f"Saved session {session.session_id} to {path}")
        except Exception as e:
            logger.error(f"Error saving session {session.session_id}: {str(e)}")
            
    def _load_sessions(self) -> None:
        """Load all sessions from the session directory."""
        if not os.path.exists(self.session_dir):
            return
            
        for filename in os.listdir(self.session_dir):
            if not filename.endswith('.json'):
                continue
                
            path = os.path.join(self.session_dir, filename)
            try:
                with open(path, 'r') as f:
                    data = json.load(f)
                session = UserSession.from_dict(data)
                self.sessions[session.session_id] = session
                logger.debug(f"Loaded session {session.session_id} from {path}")
            except Exception as e:
                logger.error(f"Error loading session from {path}: {str(e)}")
        
        logger.info(f"Loaded {len(self.sessions)} sessions from {self.session_dir}")