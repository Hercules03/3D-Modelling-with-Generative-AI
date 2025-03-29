import os
import json
import datetime

class LLMPromptLogger:
    """Logger for LLM prompt-response pairs"""
    def __init__(self):
        self.log_dir = "conversation_logs"
        self.metadata_file = os.path.join(self.log_dir, "metadata_extraction.json")
        self.category_file = os.path.join(self.log_dir, "category_analysis.json")
        self._init_log_files()

    def _init_log_files(self):
        """Initialize log files if they don't exist"""
        os.makedirs(self.log_dir, exist_ok=True)
        
        # Initialize each file with empty array if it doesn't exist
        for file_path in [self.metadata_file, self.category_file]:
            if not os.path.exists(file_path):
                with open(file_path, 'w') as f:
                    json.dump([], f, indent=2)

    def log_metadata_extraction(self, query: str, code: str, response: dict, timestamp: str = None):
        """Log metadata extraction prompt-response pair"""
        if timestamp is None:
            timestamp = datetime.datetime.now().isoformat()
            
        entry = {
            "timestamp": timestamp,
            "input": {
                "description": query,
                "code": code
            },
            "output": response,
            "tokens": {
                "input_tokens": len(query.split()) + len(code.split()),
                "output_tokens": sum(len(str(v).split()) for v in response.values())
            }
        }
        
        self._append_to_json(self.metadata_file, entry)
        print(f"Logged metadata extraction")

    def log_category_analysis(self, query: str, code: str, response: dict, timestamp: str = None):
        """Log category analysis prompt-response pair"""
        if timestamp is None:
            timestamp = datetime.datetime.now().isoformat()
            
        entry = {
            "timestamp": timestamp,
            "input": {
                "description": query,
                "code": code
            },
            "output": response,
            "tokens": {
                "input_tokens": len(query.split()) + len(code.split()),
                "output_tokens": sum(len(str(v).split()) for v in response.values())
            }
        }
        
        self._append_to_json(self.category_file, entry)
        print(f"Logged category analysis: {json.dumps(entry, indent=2)}")

    def _append_to_json(self, file_path: str, new_data: dict):
        """Append new data to existing JSON file"""
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
            
            data.append(new_data)
            
            with open(file_path, 'w') as f:
                json.dump(data, f, indent=2)
                
        except Exception as e:
            print(f"Error logging to {os.path.basename(file_path)}: {str(e)}") 