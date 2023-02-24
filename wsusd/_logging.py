import logging as logging

logging.basicConfig(format='%(levelname)s: %(message)s', level=logging.INFO)


def get_logger(name):
    return logging.getLogger(name)


def set_debug_level(logger):
    logger.setLevel(logging.DEBUG)
