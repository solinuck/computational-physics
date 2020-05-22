import sys

from pathlib import Path
import logging
from logging.handlers import TimedRotatingFileHandler

from mdengine import MDEngine


class lennardJones:
    def __init__(self, eps, sig):
        self.eps = eps
        self.sig = sig

    def force(self, r):
        return 24 * self.eps / r * (2 * (self.sig / r) ** 12 - (self.sig / r) ** 6)

    def pot(self, r):
        return 4 * self.eps * ((self.sig / r) ** 12 - (self.sig / r) ** 6)


class Logging:
    def __init__(self):
        pass

    def get_console_handler(self, formatter):
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        return console_handler

    def get_file_handler(self, log_file, formatter):
        file_handler = TimedRotatingFileHandler(self.log_file, when="midnight")
        file_handler.setFormatter(formatter)
        return file_handler

    def get_logger(self, logger_name, log_file, formatter):
        logger = logging.getLogger(logger_name)
        logger.setLevel(logging.DEBUG)
        logger.addHandler(self.get_console_handler(formatter))
        from IPython import embed

        embed()
        Path(f"/logs/{log_file}").mkdir(parents=True, exist_ok=True)
        logger.addHandler(self.get_file_handler(log_file, formatter))
        logger.propagate = False
        return logger


if __name__ == "__main__":
    logs = Logging()
    lj = lennardJones(eps=1.65e-21, sig=3.4e-10)
    engine = MDEngine(d=2, n=5, m=1, l=2, tau=1, potential=lj)
    engine.initialize()
    engine.equilibrate()

    energy_formater = f"{engine.temp} - {engine.ekin} - {engine.epot} - {engine.etot}"
    my_logger = logs.get_logger("eq_engery", "equilibration.e", energy_formater)
    my_logger.info("")
