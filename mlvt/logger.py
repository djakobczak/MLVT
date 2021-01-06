import logging
import time


CONF = {
    'logger_path': './logs/dral-{}.log'
}


class Logger:
    LOG = None

    @staticmethod
    def create_logger(name, level=logging.INFO):
        # Create a custom logger
        logger = logging.getLogger(name)
        logger.setLevel(level)

        # Create handlers
        c_handler = logging.StreamHandler()
        f_handler = logging.FileHandler(
            CONF['logger_path'].format(int(time.time())))
        c_handler.setLevel(level)
        f_handler.setLevel(level)

        # Create formatters and add it to handlers
        c_format = logging.Formatter('%(name)s - %(levelname)s - %(message)s')
        f_format = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        c_handler.setFormatter(c_format)
        f_handler.setFormatter(f_format)

        # Add handlers to the logger
        logger.addHandler(c_handler)
        logger.addHandler(f_handler)

        logger.info('Logger created')
        return logger
