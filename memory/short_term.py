"""
File: memory/short_term.py
Description: Implements the ShortTermMemory class for managing, updating, and
             pruning short-term memory.
"""

import json
from datetime import datetime
from utils.logging_config import setup_logging
from config.config import Config

logger = setup_logging()

class ShortTermMemory:
    """
    Manages short-term memory for the agent, storing recent events and perceptions.
    """
    def __init__(self, retention_threshold=None, saved_file=None):
        """
        Initializes short-term memory with a retention policy.

        Args:
            retention_threshold (int, optional): Time in seconds to retain memory entries.
            saved_file (str, optional): Path to load existing memory from a file.
        """
        self.memory = []
        self.retention_threshold = retention_threshold or Config.STM_RETENTION_THRESHOLD

        if saved_file:
            self.load(saved_file)

    def update(self, memory_entry):
        """
        Adds a new memory entry.

        Args:
            memory_entry (dict): The new memory entry to add.
        """
        self.memory.append(memory_entry)

    def prune(self, current_time=None):
        """
        Removes memory entries older than the retention threshold.

        Args:
            current_time (datetime, optional): Current time for pruning.
        """
        current_time = current_time or datetime.now().astimezone()
        logger.debug(f"Pruning at {current_time} with threshold {self.retention_threshold}s")
            
        self.memory = [
            entry for entry in self.memory
            if (current_time - datetime.fromisoformat(entry["timestamp"])).total_seconds() 
            <= self.retention_threshold
        ]

        logger.debug(f"Remaining entries: {len(self.memory)}")

    def get_recent(self, time_window=60):
        """
        Retrieves memory entries within the specified time window.

        Args:
            time_window (int): Time window in seconds.

        Returns:
            list: Recent memory entries.
        """
        current_time = datetime.now().astimezone()
        return [
            entry for entry in self.memory
            if (current_time - entry["timestamp"]).total_seconds() <= time_window
        ]

    def save(self, file_path):
        """
        Saves the memory to a file.

        Args:
            file_path (str): Path to save the memory.
        """
        with open(file_path, "w") as f:
            json.dump(self.memory, f, indent=4)

    def load(self, file_path):
        """
        Loads memory from a file.

        Args:
            file_path (str): Path to load the memory from.
        """
        try:
            with open(file_path, "r") as f:
                self.memory = json.load(f)
        except FileNotFoundError:
            logger.warning(f"File not found: {file_path}. Starting with empty memory.")
