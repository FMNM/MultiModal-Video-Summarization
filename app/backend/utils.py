import functools
import time
import hashlib
import logging
import json

logger = logging.getLogger(__name__)

# Function to check how much time consume
def time_check_decorator(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        elapsed_time = end_time - start_time

        logger.info(f"\n\t=>Function '{func.__name__}' executed in {elapsed_time:.4f} seconds.")
        return result
    return wrapper

# Function to save data to a file
def save_data(data, filepath):
    if isinstance(data, (dict, list)):
        json_data = json.dumps(data, indent=4)
        if not filepath.suffix == '.json':
            filepath.with_suffix(".json")
        with open(filepath, 'w') as file:
            file.write(json_data)
        print(f"Dictionary or List successfully saved to {filepath} as JSON.")
    elif isinstance(data, str):
        if not filepath.suffix == '.txt':
            filepath.with_suffix('.txt')
        with open(filepath, 'w') as file:
            file.write(data)
        print(f"String successfully saved to {filepath}.")
    else:
        raise ValueError("Input must be a dictionary, list, or string.")

# Function to load data from a file
def load_data(filepath):
    try:
        if filepath.suffix == '.json':
            with open(filepath, 'r') as file:
                data = json.load(file)
            print(f"JSON data successfully loaded from {filepath}")
            return data
        elif filepath.suffix == '.txt':
            with open(filepath, 'r') as file:
                data = file.read()
            print(f"Text data successfully loaded from {filepath}")
            return data
        else:
            raise ValueError("Unsupported file type. Only .json and .txt are supported.")
    except FileNotFoundError:
        print(f"The file {filepath} does not exist.")
    except Exception as e:
        print(f"An error occurred while loading the file: {str(e)}")

# Generate a SHA-256 hash from string
def sha256(str):
    hash_object = hashlib.sha256(str.encode())
    video_file_name = hash_object.hexdigest()
    return video_file_name
