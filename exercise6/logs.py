# -*- coding: utf-8 -*-

import sys

import logging

# import os
from logging.handlers import RotatingFileHandler


class Logging:
    def __init__(self, logger_name, file_name, console=True, file=True):
        self.get_logger(logger_name, file_name, console, file)

    def get_console_handler(self):
        console_handler = logging.StreamHandler(sys.stdout)
        return console_handler

    def get_file_handler(self, log_file):
        file_handler = RotatingFileHandler(log_file, backupCount=5)
        return file_handler

    def get_logger(self, logger_name, log_file, console, file):
        self.logger = logging.getLogger(logger_name)
        self.logger.setLevel(logging.DEBUG)
        if console:
            self.logger.addHandler(self.get_console_handler())
        if file:
            self.logger.addHandler(self.get_file_handler(log_file))
        self.logger.propagate = False

    def format_log(self, *args, format_nums=False):
        if format_nums:
            args = [self.num_formater(x) for x in (args)]

        text = "\t\t".join([str(x) for x in args])
        self.logger.info(text)

    @staticmethod
    def num_formater(num):
        if abs(num) > 1e3:
            format = "{:.1e}"
        else:
            format = "{:<0.2f}"
        return format.format(num)
