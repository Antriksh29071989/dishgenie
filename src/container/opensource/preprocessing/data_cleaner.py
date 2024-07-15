import logging

from langchain_text_splitters import RecursiveCharacterTextSplitter
from spacy.lang.en import English

logger = logging.getLogger(__name__)


def data_preprocessing(pages_and_texts):
    """
    Perform data preprocessing with sentencizer.
    :param pages_and_texts: List of dictionaries containing page text data
    :return: List of dictionaries with sentences extracted
    """
    try:
        nlp = English()
        nlp.add_pipe("sentencizer")
        for item in pages_and_texts:
            item["sentences"] = list(nlp(item["text"]).sents)
            item["sentences"] = [str(sentence) for sentence in item["sentences"]]
            item["page_sentence_count_spacy"] = len(item["sentences"])
        return pages_and_texts
    except Exception as e:
        logger.error(f"Failed during data preprocessing: {e}")
        raise e


def convert_to_chunks(pages_and_texts):
    """
    Convert data into chunks.
    :param pages_and_texts: List of dictionaries containing page text data
    :return: List of dictionaries with text split into chunks
    """
    try:
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1536,  # 384tokens×4characters/token=1536characters
            chunk_overlap=40,
            separators=[" "]
        )

        for item in pages_and_texts:
            joined_text = " ".join(item['sentences'])
            item["sentence_chunk"] = text_splitter.split_text(joined_text)
            item["num_chunks"] = len(item["sentence_chunk"])
        return pages_and_texts
    except Exception as e:
        logger.error(f"Failed during text chunk conversion: {e}")
        raise e
