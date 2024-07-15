import logging

from langchain_openai import ChatOpenAI

logger = logging.getLogger(__name__)


def load_model(model_name: str, temperature: int):
    logger.info("load model...")
    """
    :return: GPt model
    """
    logger.info("loading GPT model..")
    return ChatOpenAI(model=model_name, temperature=temperature)
