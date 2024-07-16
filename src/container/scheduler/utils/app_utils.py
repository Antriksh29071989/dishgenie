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

