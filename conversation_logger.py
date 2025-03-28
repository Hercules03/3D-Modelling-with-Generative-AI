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
        self.keyword_extraction_file = os.path.join(self.log_dir, "keyword_extraction_pairs.json")
        self._init_log_files()
        print("=== Conversation Logger Initialized ===\n")
    
    def _init_log_files(self):
        """Initialize log files if they don't exist"""
        print("Initializing log files...")
        os.makedirs(self.log_dir, exist_ok=True)
        print(f"- Using log directory: {self.log_dir}")
        
        # Initialize each file with empty array if it doesn't exist
        for file_path in [self.step_back_file, self.scad_gen_file, self.validation_file, self.keyword_extraction_file]:
            if not os.path.exists(file_path):
                with open(file_path, 'w') as f:
                    json.dump([], f)
                print(f"- Created new log file: {os.path.basename(file_path)}")
            else:
                print(f"- Using existing log file: {os.path.basename(file_path)}")
    
    def _append_to_json(self, file_path, new_data):
        """Append new data to existing JSON file"""
        print(f"\nAppending to file: {file_path}")
        # print(f"New data to append: {json.dumps(new_data, indent=2)}")
        try:
            # Read existing data
            with open(file_path, 'r') as f:
                data = json.load(f)
            print(f"Successfully read existing data. Current entries: {len(data)}")
            
            # Append new data
            data.append(new_data)
            print(f"Appended new data. New total entries: {len(data)}")
            
            # Write back to file
            with open(file_path, 'w') as f:
                json.dump(data, f, indent=2)
            print("Successfully wrote updated data back to file")
                
        except Exception as e:
            print(f"Error in _append_to_json: {str(e)}")
            print(f"File path: {file_path}")
            print(f"Stack trace:")
            import traceback
            traceback.print_exc()
    
    def log_step_back(self, query, analysis):
        """Log step-back analysis conversation"""
        print("\n=== Logging Step-Back Analysis ===")
        conversation = {
            'original_query': analysis.get('original_query', query),
            'focused_description': analysis.get('focused_description', query),
            'step_back_analysis': {
                'principles': analysis.get('principles', []),
                'abstractions': analysis.get('abstractions', []),
                'approach': analysis.get('approach', [])
            }
        }
        self._append_to_json(self.step_back_file, conversation)
        print("=== Step-Back Analysis Logged ===\n")
        
    def log_step_back_analysis(self, step_back_data):
        """Log user-approved step-back analysis
        
        Args:
            step_back_data (dict): Dictionary containing:
                - query: Dict with input and timestamp
                - response: Dict with principles, abstractions, approach
                - metadata: Dict with success, error, and user_approved info
        """
        print("\n=== Logging Step-Back Analysis with User Approval ===")
        try:
            # Add timestamp if not present
            if 'query' in step_back_data and 'timestamp' not in step_back_data['query']:
                step_back_data['query']['timestamp'] = datetime.now().isoformat()
                
            # Structure for logging
            entry = {
                'timestamp': datetime.now().isoformat(),
                'type': 'step_back_analysis',
                'user_approved': step_back_data.get('metadata', {}).get('user_approved', False),
                'original_query': step_back_data.get('query', {}).get('input', ''),
                'step_back_analysis': step_back_data.get('response', {})
            }
            
            self._append_to_json(self.step_back_file, entry)
            print(f"Successfully logged step-back analysis with user approval")
        except Exception as e:
            print(f"Error logging step-back analysis: {str(e)}")
            print("Stack trace:")
            import traceback
            traceback.print_exc()
        print("=== Step-Back Analysis with User Approval Logging Complete ===\n")
    
    def log_scad_generation(self, full_prompt, scad_code):
        """Log OpenSCAD generation conversation"""
        print("\n=== Logging SCAD Generation ===")
        try:
            conversation = {
                'timestamp': datetime.now().isoformat(),
                'prompt': full_prompt,
                'scad_code': scad_code,
                'type': 'scad_generation',
                'user_accepted': True  # Mark as user-accepted since this is called when user approves
            }
            
            # Extract description from the prompt
            description = ""
            for line in full_prompt.split('\n'):
                if line.startswith("USER REQUEST:"):
                    # Get the next non-empty line
                    parts = full_prompt.split("USER REQUEST:", 1)
                    if len(parts) > 1:
                        description_part = parts[1].split("\n\n", 1)[0].strip()
                        if description_part:
                            description = description_part
                    break
            
            if description:
                conversation['request'] = description
            
            print(f"SCAD Generation Entry:")
            print(f"- Timestamp: {conversation['timestamp']}")
            print(f"- Request: {conversation.get('request', '[No description extracted]')[:50]}...")
            print(f"- Code length: {len(scad_code)} characters")
            
            self._append_to_json(self.scad_gen_file, conversation)
            print("Successfully logged SCAD generation")
        except Exception as e:
            print(f"Error logging SCAD generation: {str(e)}")
            print("Stack trace:")
            import traceback
            traceback.print_exc()
        print("=== SCAD Generation Logging Complete ===\n")
    
    def log_validation(self, validation_data):
        """Log a validation decision"""
        try:
            self._append_to_json(self.validation_file, validation_data)
        except Exception as e:
            print(f"Error logging validation: {str(e)}")
    
    def log_keyword_extraction(self, query_response_pair):
        """Log keyword extraction query-response pair for fine-tuning
        
        Args:
            query_response_pair (dict): Dictionary containing:
                - query: Dict with input and timestamp
                - response: Dict with core_type, modifiers, compound_type
                - metadata: Dict with success and error info
        """
        print("\n=== Logging Keyword Extraction ===")
        print(f"Log directory: {self.log_dir}")
        print(f"Log file path: {self.keyword_extraction_file}")
        
        try:
            print(f"Received query-response pair")
            
            # Read existing data
            print("Reading existing data...")
            with open(self.keyword_extraction_file, 'r') as f:
                try:
                    data = json.load(f)
                    print(f"Successfully read existing data. Current entries: {len(data)}")
                except json.JSONDecodeError:
                    print("Error reading file - resetting to empty array")
                    data = []
            
            # Append new data
            data.append(query_response_pair)
            print(f"Added new entry. New total entries: {len(data)}")
            
            # Write back to file
            print("Writing updated data back to file...")
            with open(self.keyword_extraction_file, 'w') as f:
                json.dump(data, f, indent=2)
            print("Successfully wrote data to file")
            
            print(f"Successfully logged keyword extraction for query: {query_response_pair['query']['input']}")
        except Exception as e:
            print(f"Error logging keyword extraction: {str(e)}")
            print("Stack trace:")
            import traceback
            traceback.print_exc()
        print("=== Keyword Extraction Logging Complete ===\n")
    
    def get_scad_generation_logs(self):
        """Read and return all SCAD generation logs"""
        print("\n=== Reading SCAD Generation Logs ===")
        try:
            with open(self.scad_gen_file, 'r') as f:
                logs = json.load(f)
                # Convert the logs to the format expected by the knowledge base
                formatted_logs = []
                for log in logs:
                    # Only include entries marked as user_accepted
                    if log.get('user_accepted', False):
                        timestamp = log.get('timestamp', datetime.now().isoformat())
                        # Get request from the log or extract from prompt if needed
                        request = log.get('request', '')
                        if not request and 'prompt' in log:
                            # Try to extract description from prompt
                            prompt = log['prompt']
                            parts = prompt.split("USER REQUEST:", 1)
                            if len(parts) > 1:
                                request = parts[1].split("\n\n", 1)[0].strip()
                        
                        # Create a properly formatted log entry
                        entry = {
                            'request': request,  # The description/query
                            'code': log.get('scad_code', ''),   # The generated SCAD code
                            'timestamp': timestamp,
                            'type': 'scad_generation',
                            'user_accepted': True
                        }
                        formatted_logs.append(entry)
                        print(f"- Found user-accepted example from {timestamp}")
                
                print(f"- Successfully read {len(formatted_logs)} user-accepted SCAD generation logs out of {len(logs)} total")
                print("=== SCAD Generation Logs Read Complete ===\n")
                return formatted_logs
        except (FileNotFoundError, json.JSONDecodeError) as e:
            print(f"Error reading SCAD generation logs: {e}")
            print("=== SCAD Generation Logs Read Failed ===\n")
            return []
    
    def get_keyword_extraction_logs(self):
        """Read and return all keyword extraction logs"""
        print("\n=== Reading Keyword Extraction Logs ===")
        try:
            with open(self.keyword_extraction_file, 'r') as f:
                logs = json.load(f)
                print(f"- Successfully read {len(logs)} keyword extraction logs")
                print("=== Keyword Extraction Logs Read Complete ===\n")
                return logs
        except (FileNotFoundError, json.JSONDecodeError) as e:
            print(f"Error reading keyword extraction logs: {e}")
            print("=== Keyword Extraction Logs Read Failed ===\n")
            return []