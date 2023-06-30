import datetime
import os


def log_dir():
    """Returns the log directory.

    Note:
        Keep it simple for now
    """
    return os.environ.get("LOG_DIR", "./logs")


def create_timestamp_path(directory: str):
    """Returns a timestamped path for logging."""
    return os.path.join(directory, datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))

def save_create_directory(path):
    if not os.path.exists(path):
        os.makedirs(path)
