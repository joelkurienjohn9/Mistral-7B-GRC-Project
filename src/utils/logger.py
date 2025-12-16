"""Unified logging configuration for the project using loguru."""

import sys
from pathlib import Path
from loguru import logger


def setup_logger(
    log_file: str = None,
    level: str = "INFO",
    rotation: str = "100 MB",
    retention: str = "10 days",
    colorize: bool = True,
    backtrace: bool = True,
    diagnose: bool = True,
):
    """
    Configure the logger with consistent formatting and handlers.
    
    Args:
        log_file: Path to log file. If None, only logs to console
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        rotation: When to rotate log file (e.g., "100 MB", "1 day")
        retention: How long to keep old log files
        colorize: Whether to colorize console output
        backtrace: Whether to show backtrace on errors
        diagnose: Whether to show variable values in traceback
    
    Returns:
        Configured logger instance
    """
    # Remove default handler
    logger.remove()
    
    # Console handler with colors
    logger.add(
        sys.stderr,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
        level=level,
        colorize=colorize,
        backtrace=backtrace,
        diagnose=diagnose,
    )
    
    # File handler (if specified)
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        logger.add(
            log_file,
            format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}",
            level=level,
            rotation=rotation,
            retention=retention,
            compression="zip",
            backtrace=backtrace,
            diagnose=diagnose,
        )
        logger.info(f"Logging to file: {log_file}")
    
    return logger


def get_logger():
    """
    Get the configured logger instance.
    
    Returns:
        Logger instance
    """
    return logger

