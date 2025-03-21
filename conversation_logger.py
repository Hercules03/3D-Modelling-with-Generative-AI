import json
import os
from datetime import datetime

class ConversationLogger:
    def __init__(self):
        """Initialize the conversation logger"""
        print("\n=== Initializing Conversation Logger ===")
        self.log_dir = "conversation_logs"
        self.step_back_file = os.path.join(self.log_dir, "step_back_conversations.json")
        self.scad_gen_file = os.path.join(self.log_dir, "scad_generation_conversations.json")
        self.validation_file = os.path.join(self.log_dir, "validation_conversations.json")
        self._init_log_files()
        print("=== Conversation Logger Initialized ===\n")
    
    def _init_log_files(self):
        """Initialize log files if they don't exist"""
        print("Initializing log files...")
        os.makedirs(self.log_dir, exist_ok=True)
        print(f"- Using log directory: {self.log_dir}")
        
        # Initialize each file with empty array if it doesn't exist
        for file_path in [self.step_back_file, self.scad_gen_file, self.validation_file]:
            if not os.path.exists(file_path):
                with open(file_path, 'w') as f:
                    json.dump([], f)
                print(f"- Created new log file: {os.path.basename(file_path)}")
            else:
                print(f"- Using existing log file: {os.path.basename(file_path)}")
    
    def _append_to_json(self, file_path, new_data):
        """Append new data to existing JSON file"""
        try:
            # Read existing data
            with open(file_path, 'r') as f:
                data = json.load(f)
            
            # Append new data
            data.append(new_data)
            
            # Write back to file
            with open(file_path, 'w') as f:
                json.dump(data, f, indent=2)
                
        except Exception as e:
            print(f"Error logging to {os.path.basename(file_path)}: {e}")
    
    def log_step_back(self, query, analysis):
        """Log step-back analysis conversation"""
        print("\n=== Logging Step-Back Analysis ===")
        conversation = {
            'query': query,
            'step_back_analysis': {
                'principles': analysis.get('principles', ''),
                'abstractions': analysis.get('abstractions', ''),
                'approach': analysis.get('approach', '')
            }
        }
        self._append_to_json(self.step_back_file, conversation)
        print("=== Step-Back Analysis Logged ===\n")
    
    def log_scad_generation(self, full_prompt, scad_code):
        """Log OpenSCAD generation conversation"""
        conversation = {
            'prompt': full_prompt,
            'scad_code': scad_code,
        }
        self._append_to_json(self.scad_gen_file, conversation)
    
    def log_validation(self, validation_data):
        """Log a validation decision"""
        try:
            self._append_to_json(self.validation_file, validation_data)
        except Exception as e:
            print(f"Error logging validation: {str(e)}")
    
    def get_scad_generation_logs(self):
        """Read and return all SCAD generation logs"""
        print("\n=== Reading SCAD Generation Logs ===")
        try:
            with open(self.scad_gen_file, 'r') as f:
                logs = json.load(f)
                # Convert the logs to the format expected by the knowledge base
                formatted_logs = []
                for log in logs:
                    formatted_logs.append({
                        'request': log.get('prompt', ''),  # The original prompt/request
                        'code': log.get('scad_code', '')   # The generated SCAD code
                    })
                print(f"- Successfully read {len(formatted_logs)} SCAD generation logs")
                print("=== SCAD Generation Logs Read Complete ===\n")
                return formatted_logs
        except (FileNotFoundError, json.JSONDecodeError) as e:
            print(f"Error reading SCAD generation logs: {e}")
            print("=== SCAD Generation Logs Read Failed ===\n")
            return []