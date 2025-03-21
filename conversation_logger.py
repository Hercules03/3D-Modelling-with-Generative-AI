import json
import os

class ConversationLogger:
    def __init__(self):
        """Initialize the conversation logger"""
        self.log_dir = "conversation_logs"
        self.step_back_file = os.path.join(self.log_dir, "step_back_conversations.json")
        self.scad_gen_file = os.path.join(self.log_dir, "scad_generation_conversations.json")
        self.validation_file = os.path.join(self.log_dir, "validation_conversations.json")
        self._init_log_files()
    
    def _init_log_files(self):
        """Initialize log files if they don't exist"""
        os.makedirs(self.log_dir, exist_ok=True)
        
        # Initialize each file with empty array if it doesn't exist
        for file_path in [self.step_back_file, self.scad_gen_file, self.validation_file]:
            if not os.path.exists(file_path):
                with open(file_path, 'w') as f:
                    json.dump([], f)
    
    def _append_to_json(self, file_path, new_data):
        """Append new data to existing JSON file"""
        try:
            # Read existing data
            with open(file_path, 'r') as f:
                data = json.load(f)
            
            # Append new data without timestamp
            data.append(new_data)
            
            # Write back to file
            with open(file_path, 'w') as f:
                json.dump(data, f, indent=2)
                
        except Exception as e:
            print(f"Error logging conversation: {e}")
    
    def log_step_back(self, query, analysis):
        """Log step-back analysis conversation"""
        conversation = {
            'query': query,
            'step_back_analysis': {
                'principles': analysis.get('principles', ''),
                'abstractions': analysis.get('abstractions', ''),
                'approach': analysis.get('approach', '')
            }
        }
        self._append_to_json(self.step_back_file, conversation)
    
    def log_scad_generation(self, full_prompt, scad_code):
        """Log OpenSCAD generation conversation"""
        conversation = {
            'prompt': full_prompt,
            'scad_code': scad_code
        }
        self._append_to_json(self.scad_gen_file, conversation)
    
    def log_validation(self, desired_object, example_object, decision):
        """Log validation conversation"""
        conversation = {
            'desired_object': desired_object,
            'example_object': example_object,
            'decision': decision  # 'USEFUL' or 'UNUSEFUL'
        }
        self._append_to_json(self.validation_file, conversation)