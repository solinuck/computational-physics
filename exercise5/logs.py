import sys

import logging
from logging.handlers import TimedRotatingFileHandler


class Logging:
    def __init__(self):
        self.formatter = logging.Formatter("%(message)s")

    def get_console_handler(self):
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(self.formatter)
        return console_handler

    def get_file_handler(self, log_file):
        file_handler = TimedRotatingFileHandler(log_file, when="midnight")
        file_handler.setFormatter(self.formatter)
        return file_handler

    def get_logger(self, logger_name, log_file):
        logger = logging.getLogger(logger_name)
        logger.setLevel(logging.DEBUG)
        logger.addHandler(self.get_console_handler())
        logger.addHandler(self.get_file_handler(log_file))
        logger.propagate = False
        return logger
