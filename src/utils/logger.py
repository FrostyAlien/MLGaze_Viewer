"""Centralized logging utility for MLGaze Viewer."""

import logging
import sys
from pathlib import Path
from datetime import datetime
from typing import Optional


class LoggerWrapper:
    """Wrapper for logger instances to provide custom methods."""
    
    def __init__(self, logger):
        """Initialize wrapper with a logger instance."""
        self._logger = logger
    
    def info(self, message: str, *args, **kwargs):
        """Log info message."""
        self._logger.info(message, *args, **kwargs)
    
    def warning(self, message: str, *args, **kwargs):
        """Log warning message."""
        self._logger.warning(message, *args, **kwargs)
    
    def error(self, message: str, *args, **kwargs):
        """Log error message."""
        self._logger.error(message, *args, **kwargs)
    
    def debug(self, message: str, *args, **kwargs):
        """Log debug message."""
        self._logger.debug(message, *args, **kwargs)
    
    def success(self, message: str, *args, **kwargs):
        """Log success message (as INFO with special prefix)."""
        self._logger.info(f"✓ {message}", *args, **kwargs)


class ColoredFormatter(logging.Formatter):
    """Custom formatter with colors for console output."""
    
    COLORS = {
        'DEBUG': '\033[36m',     # Cyan
        'INFO': '\033[32m',      # Green
        'WARNING': '\033[33m',   # Yellow
        'ERROR': '\033[31m',     # Red
        'CRITICAL': '\033[35m',  # Magenta
        'RESET': '\033[0m'       # Reset
    }
    
    def format(self, record):
        """Format the log record with colors."""
        log_color = self.COLORS.get(record.levelname, self.COLORS['RESET'])
        record.levelname = f"{log_color}{record.levelname}{self.COLORS['RESET']}"
        return super().format(record)


class MLGazeLogger:
    """Singleton logger for MLGaze Viewer application."""
    
    _instance = None
    _logger = None
    
    def __new__(cls):
        """Ensure only one instance exists."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        """Initialize the logger if not already done."""
        if self._logger is None:
            self._setup_logger()
    
    def _setup_logger(self):
        """Configure the logger with console and file handlers."""
        self._logger = logging.getLogger('MLGazeViewer')
        self._logger.setLevel(logging.DEBUG)
        
        # Remove any existing handlers
        self._logger.handlers.clear()
        
        # Console handler with colors
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)
        console_format = ColoredFormatter(
            '%(asctime)s - %(levelname)s - %(name)s - %(message)s'
        )
        console_handler.setFormatter(console_format)
        self._logger.addHandler(console_handler)
        
        # File handler (optional, created in logs directory)
        log_dir = Path('logs')
        if not log_dir.exists():
            try:
                log_dir.mkdir(parents=True, exist_ok=True)
            except Exception:
                # If we can't create logs directory, skip file logging
                return
        
        log_file = log_dir / f"mlgaze_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.DEBUG)
        file_format = logging.Formatter(
            '%(asctime)s - %(levelname)s - %(name)s.%(funcName)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        file_handler.setFormatter(file_format)
        self._logger.addHandler(file_handler)
    
    def get_logger(self, name: Optional[str] = None):
        """Get a logger instance.
        
        Args:
            name: Optional name for the logger (e.g., module name)
            
        Returns:
            Logger wrapper with custom methods
        """
        if name:
            child_logger = self._logger.getChild(name)
            return LoggerWrapper(child_logger)
        return LoggerWrapper(self._logger)
    
    def info(self, message: str, *args, **kwargs):
        """Log info message."""
        self._logger.info(message, *args, **kwargs)
    
    def warning(self, message: str, *args, **kwargs):
        """Log warning message."""
        self._logger.warning(message, *args, **kwargs)
    
    def error(self, message: str, *args, **kwargs):
        """Log error message."""
        self._logger.error(message, *args, **kwargs)
    
    def debug(self, message: str, *args, **kwargs):
        """Log debug message."""
        self._logger.debug(message, *args, **kwargs)
    
    def success(self, message: str, *args, **kwargs):
        """Log success message (as INFO with special prefix)."""
        self._logger.info(f"✓ {message}", *args, **kwargs)


# Global logger instance
logger = MLGazeLogger()