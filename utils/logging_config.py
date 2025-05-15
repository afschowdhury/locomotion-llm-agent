"""
File: utils/logging_config.py
Description: Logging configuration for the agent workflow.
"""

import logging
import os
import sys
from datetime import datetime

def setup_logging(logger_name='locomotion', log_dir="logs", console_level=logging.INFO, session_id=None):
    """
    Enhanced logging configuration with rotation and multiple handlers
    
    Args:
        logger_name: Name of the logger
        log_dir: Directory to store log files
        console_level: Minimum logging level for console output
        session_id: Unique identifier for the test session
    """
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    logger = logging.getLogger(logger_name)
    
    if not logger.handlers or (session_id and not hasattr(logger, 'session_id') or logger.session_id != session_id):
        logger.handlers.clear()
        logger.setLevel(logging.DEBUG)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{log_dir}/{logger_name}_{session_id or timestamp}.log"
        
        file_handler = logging.FileHandler(
            filename=filename,
            mode='a',
            encoding='utf-8'
        )
        file_handler.setLevel(logging.DEBUG)
        
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(console_level)

        file_formatter = logging.Formatter(
            '%(asctime)s | %(levelname)-8s | %(name)-12s | %(message).10000s'
        )
        console_formatter = logging.Formatter(
            '%(levelname)-8s | %(module)-15s | %(message)s'
        )
        file_handler.setFormatter(file_formatter)
        console_handler.setFormatter(console_formatter)

        logger.addHandler(file_handler)
        logger.addHandler(console_handler)
        
        logger.session_id = session_id
        
    logging.captureWarnings(True)
    return logger
