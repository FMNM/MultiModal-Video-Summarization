import os
import logging
import json


# Get the logging level from the environment variable
log_level = os.getenv("LOGGING_LEVEL", "INFO").upper()
print(f"LOGGING_LEVEL:{log_level}")
numeric_level = getattr(logging, log_level, logging.INFO)
root_logger_level = numeric_level

# first of all root logger is INFO
logging.basicConfig(level=root_logger_level)


# Dictionary to store all loggers
all_loggers = {}

app_logger_level = logging.INFO


# Get the logging level from the environment variable
log_level = os.getenv("APP_LOGGING_LEVEL", "INFO").upper()
print(f"APP_LOGGING_LEVEL:{log_level}")
numeric_level = getattr(logging, log_level, logging.INFO)
app_logger_level = numeric_level

def get_logger(name):
    if name not in all_loggers:
        # Create and store the logger if not already present
        logger = logging.getLogger(name)
        logger.setLevel(app_logger_level)
        all_loggers[name] = logger

        print(f"{name} : {app_logger_level} {list(all_loggers.keys())}")

    return all_loggers[name]

def set_all_loggers_level(level):
    app_logger_level = level
    for logger in all_loggers.values():
        logger.setLevel(level)
    

def get_all_loggers():
    return list(all_loggers.values())

