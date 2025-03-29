import json
import os
from datetime import datetime
from typing import Dict, Any, Optional

class LLMPromptLogger:
    """A logger for LLM prompts and responses."""
    
    def __init__(self, log_dir: str = "conversation_logs"):
        """Initialize the logger with a directory for storing logs.
        
        Args:
            log_dir: Directory to store log files. Defaults to "conversation_logs".
        """
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)
        
        # Initialize log files
        self.step_back_log = os.path.join(log_dir, "step_back_conversations.json")
        self.scad_gen_log = os.path.join(log_dir, "scad_generation_conversations.json")
        self.validation_log = os.path.join(log_dir, "validation_conversations.json")
        self.keyword_log = os.path.join(log_dir, "keyword_extraction_pairs.json")
        self.metadata_log = os.path.join(log_dir, "metadata_extraction.json")
        self.category_log = os.path.join(log_dir, "category_analysis.json")
        
        # Create log files if they don't exist
        for log_file in [self.step_back_log, self.scad_gen_log, 
                        self.validation_log, self.keyword_log,
                        self.metadata_log, self.category_log]:
            if not os.path.exists(log_file):
                with open(log_file, 'w') as f:
                    json.dump([], f)

    def _read_log(self, log_file: str) -> list:
        """Read a log file and return its contents.
        
        Args:
            log_file: Path to the log file.
            
        Returns:
            List of log entries.
        """
        try:
            with open(log_file, 'r') as f:
                return json.load(f)
        except json.JSONDecodeError:
            return []

    def _write_log(self, log_file: str, data: list) -> None:
        """Write data to a log file.
        
        Args:
            log_file: Path to the log file.
            data: List of log entries to write.
        """
        with open(log_file, 'w') as f:
            json.dump(data, f, indent=2)

    def _calculate_tokens(self, text: str) -> int:
        """Calculate approximate token count from text.
        
        Args:
            text: Text to count tokens for.
            
        Returns:
            Approximate token count.
        """
        return len(str(text).split())

    def log_step_back(self, description: str, analysis: Dict[str, Any]) -> None:
        """Log a step-back analysis conversation.
        
        Args:
            description: The input description.
            analysis: The step-back analysis output.
        """
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "input": description,
            "output": analysis
        }
        
        logs = self._read_log(self.step_back_log)
        logs.append(log_entry)
        self._write_log(self.step_back_log, logs)

    def log_scad_generation(self, description: str, code: str, metadata: Optional[Dict] = None) -> None:
        """Log a SCAD code generation conversation.
        
        Args:
            description: The input description.
            code: The generated SCAD code.
            metadata: Optional metadata about the generation.
        """
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "input": description,
            "output": code,
            "metadata": metadata or {}
        }
        
        logs = self._read_log(self.scad_gen_log)
        logs.append(log_entry)
        self._write_log(self.scad_gen_log, logs)

    def log_validation(self, code: str, validation_result: Dict[str, Any]) -> None:
        """Log a code validation conversation.
        
        Args:
            code: The SCAD code being validated.
            validation_result: The validation results.
        """
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "input": code,
            "output": validation_result
        }
        
        logs = self._read_log(self.validation_log)
        logs.append(log_entry)
        self._write_log(self.validation_log, logs)

    def log_keyword_extraction(self, description: str, keywords: Dict[str, Any]) -> None:
        """Log a keyword extraction conversation.
        
        Args:
            description: The input description.
            keywords: The extracted keywords.
        """
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "input": description,
            "output": keywords
        }
        
        logs = self._read_log(self.keyword_log)
        logs.append(log_entry)
        self._write_log(self.keyword_log, logs)

    def log_metadata_extraction(self, query: str, code: str, response: Dict[str, Any]) -> None:
        """Log metadata extraction prompt-response pair.
        
        Args:
            query: The input query.
            code: The SCAD code.
            response: The extracted metadata.
        """
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "input": {
                "description": query,
                "code": code
            },
            "output": response,
            "tokens": {
                "input_tokens": self._calculate_tokens(query) + self._calculate_tokens(code),
                "output_tokens": sum(self._calculate_tokens(v) for v in response.values())
            }
        }
        
        logs = self._read_log(self.metadata_log)
        logs.append(log_entry)
        self._write_log(self.metadata_log, logs)
        print(f"Logged metadata extraction")

    def log_category_analysis(self, query: str, code: str, response: Dict[str, Any]) -> None:
        """Log category analysis prompt-response pair.
        
        Args:
            query: The input query.
            code: The SCAD code.
            response: The category analysis results.
        """
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "input": {
                "description": query,
                "code": code
            },
            "output": response,
            "tokens": {
                "input_tokens": self._calculate_tokens(query) + self._calculate_tokens(code),
                "output_tokens": sum(self._calculate_tokens(v) for v in response.values())
            }
        }
        
        logs = self._read_log(self.category_log)
        logs.append(log_entry)
        self._write_log(self.category_log, logs)
        print(f"Logged category analysis: {json.dumps(log_entry, indent=2)}")

    def get_all_logs(self) -> Dict[str, list]:
        """Get all logs from all log files.
        
        Returns:
            Dictionary containing all logs, keyed by log type.
        """
        return {
            "step_back": self._read_log(self.step_back_log),
            "scad_generation": self._read_log(self.scad_gen_log),
            "validation": self._read_log(self.validation_log),
            "keyword_extraction": self._read_log(self.keyword_log),
            "metadata_extraction": self._read_log(self.metadata_log),
            "category_analysis": self._read_log(self.category_log)
        }

    def get_log_by_type(self, log_type: str) -> list:
        """Get logs of a specific type.
        
        Args:
            log_type: Type of log to retrieve ("step_back", "scad_generation", 
                     "validation", "keyword_extraction", "metadata_extraction",
                     or "category_analysis").
                     
        Returns:
            List of log entries for the specified type.
            
        Raises:
            ValueError: If log_type is invalid.
        """
        log_files = {
            "step_back": self.step_back_log,
            "scad_generation": self.scad_gen_log,
            "validation": self.validation_log,
            "keyword_extraction": self.keyword_log,
            "metadata_extraction": self.metadata_log,
            "category_analysis": self.category_log
        }
        
        if log_type not in log_files:
            raise ValueError(f"Invalid log type: {log_type}")
            
        return self._read_log(log_files[log_type]) 