import logging

def configure_logging(log_file):
    # Create a logger with the desired name
    logger = logging.getLogger('my_logger')

    # If the logger already has handlers, remove them
    if logger.hasHandlers():
        logger.handlers.clear()

    # Set the logger level to INFO
    logger.setLevel(logging.INFO)

    # Create a file handler that writes to the log file
    file_handler = logging.FileHandler(log_file)

    # Set the file handler level to INFO
    file_handler.setLevel(logging.INFO)

    # Create a formatter for the log messages
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    # Add the formatter to the file handler
    file_handler.setFormatter(formatter)

    # Add the file handler to the logger
    logger.addHandler(file_handler)

    return logger