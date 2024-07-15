import logging
import os

from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory

logger = logging.getLogger(__name__)


def set_gpt_key(key: str):
    """
     Setting up the key in environment
    :param key:
    """
    try:
        os.environ['OPENAI_API_KEY'] = key
    except Exception as e:
        logger.error("There is an issue found in setting up the GPT key", e)
        raise "Not able to set GPT key."


def has_embeddings_created(directory_path: str) -> bool:
    """
    Check if directory contains persisted embedding files.
    :param directory_path:
    :return: True or False
    """
    if os.path.exists(directory_path) and os.path.isdir(directory_path):
        if any(os.path.isfile(os.path.join(directory_path, f)) for f in os.listdir(directory_path)):
            return True
        else:
            return False
    else:
        print("Directory does not exist")
