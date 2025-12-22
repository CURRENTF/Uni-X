import logging
import os
from datetime import datetime


def get_time_str_with_local_rank():
    """
    Get the current time and return it in the format 'yymmdd-hhmmss'.

    :return: String of the current time in the format 'yymmdd-hhmmss'
    """
    rank = os.environ.get('LOCAL_RANK')
    return f"{datetime.now().strftime('%y%m%d-%H%M%S')}-R{rank}"
    # return f"{datetime.now().strftime('%y%m%d-%H%M%S')}"


# Create a logger and configure it
def create_logger(
        name, log_path, terminal_level=logging.INFO, file_level=logging.DEBUG
):
    """
    Create a logger that outputs logs to both the console and a log file.
    In a multi-process environment (e.g., using accelerate), file logging is only active for the main process.

    :param name: The name of the logger
    :param log_path: The path to the log file
    :param terminal_level: Log level for console output, default is INFO
    :param file_level: Log level for file output, default is DEBUG
    :return: The configured logger object
    """
    logger = logging.getLogger(name)

    # If handlers already exist, return the existing logger to avoid duplicate handlers
    if logger.hasHandlers():
        return logger

    logger.setLevel(logging.DEBUG)

    # Log format, including filename and line number
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s"
    )

    # Console handler for all processes
    console_handler = logging.StreamHandler()
    console_handler.setLevel(terminal_level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    os.makedirs(os.path.dirname(log_path), exist_ok=True)

    file_handler = logging.FileHandler(log_path)
    file_handler.setLevel(file_level)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    return logger


# Get an existing logger
def get_logger(name):
    return logging.getLogger(name)


main_logger = create_logger("main", f"./outputs/{get_time_str_with_local_rank()}.log")
