import platform
import threading
import logging

import webrtcvad

import torch
import torch.multiprocessing as mp


def start_thread(target=None, args=()):
    """
    Start a thread or process based on the operating system.

    This method initiates a thread on Linux using threading.Thread,
    and a process on other systems using multiprocessing.Process.

    Args:
        target (callable): The function to be executed by the thread or process.
                           Defaults to None, meaning no function will be called.
        args (tuple): Arguments to pass to the target function. Defaults to ().

    Returns:
        threading.Thread or multiprocessing.Process: The started thread or process object.
    """
    is_linux = platform.system() == 'Linux'
    thread_class = threading.Thread if is_linux else mp.Process

    thread = thread_class(target=target, args=args)
    if is_linux:
        thread.daemon = True  # Set as a daemon thread on Linux

    thread.start()
    return thread


def find_suffix_match_in_text(source_text, target_text, suffix_len=10):
    """
    Find the position where the last 'n' characters of text1 match with a substring in text2.

    This function extracts the last 'n' characters from text1 (where 'n' is determined by
    'length_of_match') and searches for this substring in text2, starting from the end of text2.

    Args:
        - source_text (str): The text containing the substring to find in target_text.
        - target_text (str): The text in which to search for the matching substring.
        - length_of_match (int): The length of the matching string to search for.

    Returns:
        int: The starting index (0-based) in text2 where the matching substring begins.
            Returns -1 if no match is found or if either text is too short.
    """

    # Check if either text is too short for the specified length of match
    if len(source_text) < suffix_len or len(target_text) < suffix_len:
        return -1

    # Extract the end portion of text1 to compare
    target_substring = source_text[-suffix_len:]

    # Loop through text2 from the end to the beginning to find a match
    for i in range(len(target_text) - suffix_len, -1, -1):
        # Compare the current substring with the target substring
        if target_text[i:i + suffix_len] == target_substring:
            return i

    return -1


def set_multiprocessing_start_method(method="spawn"):
    """
    Set the multiprocessing start method if it hasn't been set already.

    Args:
        method (str): The start method to set. Default is "spawn".
    """
    try:
        # Only set the start method if it hasn't been set already
        if mp.get_start_method(allow_none=True) is None:
            mp.set_start_method(method)
    except RuntimeError as e:
        logging.info(f"Start method has already been set. Details: {e}")


def set_up_webrtc(sensitivity):
    model = webrtcvad.Vad()
    model.set_mode(sensitivity)
    return model


def set_up_silero(use_onnx):
    model, _ = torch.hub.load(
        repo_or_dir="snakers4/silero-vad",
        model="silero_vad",
        verbose=False,
        onnx=use_onnx
    )
    return model


def initialize_logging(console_logging_level=logging.INFO, disable_log_file=False, log_file='sst.log'):
    """
    Initialize the logging configuration with specified settings.

    Args:
        console_logging_level (int): The logging level for console output (e.g., logging.INFO).
        disable_log_file (bool): If True, disables logging to a file. Default is False.
        log_file (str): The name of the log file. Default is 'sst.log'.
    """
    log_format = '\nReal-Time Speech to Text: %(name)s - %(levelname)s - %(message)s'
    file_log_format = '%(asctime)s.%(msecs)03d - ' + log_format

    # Get the root logger and set its level to DEBUG
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)

    # Remove any existing handlers
    logger.handlers.clear()

    # Create and configure console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(console_logging_level)
    console_handler.setFormatter(logging.Formatter(log_format))
    logger.addHandler(console_handler)

    # Create and configure file handler if logging to a file is enabled
    if not disable_log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(logging.Formatter(
            file_log_format, datefmt='%Y-%m-%d %H:%M:%S'
        ))
        logger.addHandler(file_handler)
