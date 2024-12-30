import sys

from loguru import logger


def setup_logger(log_file_path=None, stderr_level=None):
    logger.remove() # remove the default sys.stderr logger
    if log_file_path is not None:
        logger.add(log_file_path, rotation="500 MB")
    
    if stderr_level is not None:
        logger.add(sys.stderr, level=stderr_level)
    else:
        logger.add(sys.stderr, format="{time} {level} {message}", level='DEBUG')
        