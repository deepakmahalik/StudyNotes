import logging
from logging.handlers import TimedRotatingFileHandler
import os
import sys

# Create logs directory if it doesn't exist
LOGS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'logs')
if not os.path.exists(LOGS_DIR):
    os.makedirs(LOGS_DIR)

def setup_logger(name="mindmap_agent"):
    logger = logging.getLogger(name)
    
    # If the logger already has handlers, return it to avoid duplicate logs
    if logger.hasHandlers():
        return logger
        
    logger.setLevel(logging.DEBUG)

    # Log format: Timestamp - LogLevel - LoggerName - Message
    formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - [%(name)s] - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    # Datewise rotating file handler (creates a new file every midnight)
    log_file_path = os.path.join(LOGS_DIR, f"{name}.log")
    file_handler = TimedRotatingFileHandler(
        log_file_path,
        when="midnight",
        interval=1,
        backupCount=30,  # Keep logs for 30 days
        encoding='utf-8'
    )
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)
    file_handler.suffix = "%Y-%m-%d"

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)

    # Add handlers
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    # Global exception hook to capture unhandled exceptions automatically
    def handle_exception(exc_type, exc_value, exc_traceback):
        if issubclass(exc_type, KeyboardInterrupt):
            sys.__excepthook__(exc_type, exc_value, exc_traceback)
            return
        logger.error("Uncaught exception", exc_info=(exc_type, exc_value, exc_traceback))

    sys.excepthook = handle_exception

    return logger

# Provide a default logger instance
log = setup_logger()
