import sys

import logging
import os
from logging.handlers import RotatingFileHandler


class Logging:
    def __init__(self):
        self.formatter = logging.Formatter("%(message)s")

    def get_console_handler(self):
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(self.formatter)
        return console_handler

    def get_file_handler(self, log_file):
        file_handler = RotatingFileHandler(log_file, backupCount=5)
        file_handler.setFormatter(self.formatter)
        should_roll_over = os.path.isfile(log_file)
        if should_roll_over:  # log already exists, roll over!
            file_handler.doRollover()
        return file_handler

    def get_logger(self, logger_name, log_file):
        logger = logging.getLogger(logger_name)
        logger.setLevel(logging.DEBUG)
        logger.addHandler(self.get_console_handler())
        logger.addHandler(self.get_file_handler(log_file))
        logger.propagate = False
        return logger
