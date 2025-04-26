import logging
import sys
from typing import Optional

"""
Author: Fernando Gallego
Affiliation: Researcher at the Computational Intelligence (ICB) Group, University of Málaga
"""

def setup_custom_logger(
    name: str,
    log_file: Optional[str] = None,
    enable_console: bool = True
) -> logging.Logger:
    """
    Set up a custom logger with optional colored output for the console.

    This logger supports both file-based logging and console output with conditional
    color formatting based on the log level.

    Args:
        name (str): Name of the logger.
        log_file (Optional[str]): Path to the log file. If not provided, logging to a file is disabled.
        enable_console (bool): If True, logs will also be printed to the console. Defaults to True.

    Returns:
        logging.Logger: Configured logger with optional color formatting and multiple outputs.

    Example:
        logger = setup_custom_logger("example_logger", log_file="example.log", enable_console=True)
        logger.info("This is an informational message.")
    """
    class CustomFormatter(logging.Formatter):
        """
        Custom logging formatter with optional color support for console output.

        Attributes:
            base_format (str): Format string for log messages.
            datefmt (str): Date format for log messages.
            COLORS (dict): ANSI color codes for different log levels.
        """
        base_format = "%(asctime)s - %(levelname)s - %(message)s (%(filename)s:%(lineno)d)"
        datefmt = "%Y-%m-%d %H:%M:%S"

        COLORS = {
            logging.DEBUG: "\x1b[38;21m",  # Light blue
            logging.INFO: "\x1b[37m",     # White
            logging.WARNING: "\x1b[33m", # Yellow
            logging.ERROR: "\x1b[31m",   # Red
            logging.CRITICAL: "\x1b[31;1m",  # Bright red
            "RESET": "\x1b[0m"           # Reset color
        }

        def format(self, record: logging.LogRecord) -> str:
            """
            Format the log record, applying color if supported.

            Args:
                record (logging.LogRecord): The log record to format.

            Returns:
                str: The formatted log message.
            """
            if self._stream_supports_color(record):
                log_fmt = self.COLORS.get(record.levelno, "RESET") + self.base_format + self.COLORS["RESET"]
            else:
                log_fmt = self.base_format
            self._style._fmt = log_fmt
            return super().format(record)

        def _stream_supports_color(self, record: logging.LogRecord) -> bool:
            """
            Determine if the stream supports color output.

            Args:
                record (logging.LogRecord): The log record to check.

            Returns:
                bool: True if the stream supports color, False otherwise.
            """
            if getattr(record, 'stream', None):
                return hasattr(record.stream, 'isatty') and record.stream.isatty()
            return False

    # Configure log handlers
    handlers = []
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(CustomFormatter())
        handlers.append(file_handler)
        enable_console = False  # Disable console output if logging to a file

    if enable_console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(CustomFormatter())
        handlers.append(console_handler)

    # Set up the logger
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)  # Log all levels from DEBUG upwards
    for handler in handlers:
        logger.addHandler(handler)

    return logger
