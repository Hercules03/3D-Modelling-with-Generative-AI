"""
Session Integration Utilities

This module provides helper functions to integrate the UserSession functionality
with the main OpenSCAD generator application.
"""

import logging
from typing import Dict, Any, Optional, Tuple

from user_session import SessionManager, UserSession

# Configure logging
logger = logging.getLogger(__name__)

class SessionIntegration:
    """
    Provides integration methods between the SessionManager and the 3D modeler application.
    """
    
    def __init__(self, session_dir: str = "user_sessions"):
        """
        Initialize the session integration.
        
        Args:
            session_dir (str, optional): Directory to store user sessions
        """
        self.session_manager = SessionManager(session_dir=session_dir)
        self.current_session_id = None
        logger.info("Session integration initialized")
        
    def start_session(self, user_id: str = None) -> Tuple[str, UserSession]:
        """
        Start a new user session.
        
        Args:
            user_id (str, optional): User identifier
            
        Returns:
            Tuple[str, UserSession]: Session ID and session object
        """
        session = self.session_manager.create_session(user_id=user_id)
        self.current_session_id = session.session_id
        logger.info(f"Started new session: {session.session_id}")
        return session.session_id, session
        
    def get_current_session(self) -> Optional[UserSession]:
        """
        Get the current active session.
        
        Returns:
            Optional[UserSession]: Current session or None if not set
        """
        if not self.current_session_id:
            return None
        return self.session_manager.get_session(self.current_session_id)
        
    def switch_session(self, session_id: str) -> Optional[UserSession]:
        """
        Switch to a different session.
        
        Args:
            session_id (str): ID of the session to switch to
            
        Returns:
            Optional[UserSession]: The switched-to session, or None if not found
        """
        session = self.session_manager.get_session(session_id)
        if session:
            self.current_session_id = session_id
            logger.info(f"Switched to session: {session_id}")
            return session
        else:
            logger.warning(f"Session not found: {session_id}")
            return None
            
    def record_generation(self, description: str, code: str, metadata: Dict[str, Any] = None) -> Optional[Dict]:
        """
        Record a generation in the current session.
        
        Args:
            description (str): The generation prompt/description
            code (str): The generated OpenSCAD code
            metadata (Dict, optional): Additional metadata
            
        Returns:
            Optional[Dict]: The generation entry, or None if no active session
        """
        session = self.get_current_session()
        if not session:
            logger.warning("No active session to record generation")
            return None
            
        entry = session.add_generation(description, code, metadata)
        self.session_manager.save_session(session)
        logger.info(f"Recorded generation {entry['id']} in session {session.session_id}")
        return entry
        
    def get_user_history(self, user_id: str = None) -> Dict[str, Any]:
        """
        Get the generation history for a user across all their sessions.
        
        Args:
            user_id (str, optional): User ID to get history for. If None, uses current session's user
            
        Returns:
            Dict: History data organized by session
        """
        # If no user_id provided, try to get from current session
        if not user_id:
            current_session = self.get_current_session()
            if not current_session:
                logger.warning("No current session and no user_id provided")
                return {}
            user_id = current_session.user_id
            
        # Get all sessions for this user
        sessions = self.session_manager.get_user_sessions(user_id)
        
        # Organize history by session
        history = {
            "user_id": user_id,
            "sessions": []
        }
        
        for session in sessions:
            session_data = {
                "session_id": session.session_id,
                "created_at": session.created_at,
                "last_active": session.last_active,
                "generations": session.get_history()
            }
            history["sessions"].append(session_data)
            
        # Sort sessions by last_active (newest first)
        history["sessions"].sort(key=lambda s: s["last_active"], reverse=True)
        
        return history
        
    def get_generation_example(self, generation_id: str, session_id: str = None) -> Optional[Dict]:
        """
        Get a specific generation example.
        
        Args:
            generation_id (str): ID of the generation to retrieve
            session_id (str, optional): Session ID to look in. If None, searches all sessions
            
        Returns:
            Optional[Dict]: The generation entry, or None if not found
        """
        # If session_id provided, only look in that session
        if session_id:
            session = self.session_manager.get_session(session_id)
            if not session:
                logger.warning(f"Session {session_id} not found")
                return None
            return session.get_generation(generation_id)
            
        # Otherwise, search all sessions
        for session in self.session_manager.sessions.values():
            entry = session.get_generation(generation_id)
            if entry:
                return entry
                
        logger.warning(f"Generation {generation_id} not found in any session")
        return None
    
def display_user_history(history):
    """Display the generation history for a user."""
    if not history or not history.get('sessions'):
        print("\nNo generation history found.")
        return
    
    user_id = history.get('user_id', 'Unknown User')
    sessions = history.get('sessions', [])
    
    print(f"\n=== Generation History for User: {user_id} ===")
    print(f"Total Sessions: {len(sessions)}")
    
    for i, session in enumerate(sessions):
        session_id = session.get('session_id', 'Unknown Session')
        created_at = session.get('created_at', 'Unknown')
        last_active = session.get('last_active', 'Unknown')
        generations = session.get('generations', [])
        
        print(f"\nSession {i+1}: {session_id}")
        print(f"  Created: {created_at}")
        print(f"  Last Active: {last_active}")
        print(f"  Generations: {len(generations)}")
        
        if generations:
            print("\n  Generation History:")
            print("  " + "-" * 40)
            
            for j, gen in enumerate(generations):
                gen_id = gen.get('id', 'Unknown')
                timestamp = gen.get('timestamp', 'Unknown')
                description = gen.get('description', 'No description')
                code_length = len(gen.get('code', ''))
                metadata = gen.get('metadata', {})
                
                print(f"  {j+1}. Generation ID: {gen_id}")
                print(f"     Time: {timestamp}")
                print(f"     Description: {description[:50]}{'...' if len(description) > 50 else ''}")
                print(f"     Code Length: {code_length} characters")
                
                # Display metadata
                if metadata:
                    print(f"     Metadata:")
                    for key, value in metadata.items():
                        if isinstance(value, list):
                            if value:
                                print(f"       - {key}: {', '.join(value[:3])}{'...' if len(value) > 3 else ''}")
                        else:
                            print(f"       - {key}: {value}")
                
                print("  " + "-" * 40)
    
    print("\nEnd of Generation History")
    
    # Ask if user wants to view full details of a specific generation
    view_details = input("\nView details of a specific generation? (y/n): ").lower().strip()
    if view_details == 'y':
        session_idx = input("Enter session number: ")
        gen_idx = input("Enter generation number: ")
        
        try:
            session_idx = int(session_idx) - 1
            gen_idx = int(gen_idx) - 1
            
            if 0 <= session_idx < len(sessions) and 0 <= gen_idx < len(sessions[session_idx].get('generations', [])):
                gen = sessions[session_idx]['generations'][gen_idx]
                
                print("\n" + "=" * 50)
                print(f"Generation Details:")
                print("=" * 50)
                print(f"ID: {gen.get('id')}")
                print(f"Timestamp: {gen.get('timestamp')}")
                print(f"Description: {gen.get('description')}")
                
                print("\nMetadata:")
                for key, value in gen.get('metadata', {}).items():
                    print(f"  {key}: {value}")
                
                print("\nOpenSCAD Code:")
                print("-" * 50)
                print(gen.get('code', 'No code available'))
                print("-" * 50)
            else:
                print("Invalid session or generation number.")
        except ValueError:
            print("Please enter valid numbers.")