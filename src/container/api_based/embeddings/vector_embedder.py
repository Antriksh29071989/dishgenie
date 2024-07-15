import logging
import os

from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

logger = logging.getLogger(__name__)


def load_data_from_vector_store(directory_path: str) -> Chroma:
    """
    :param directory_path: loading data from vector store
    :return: Chroma vector store.
    """
    logger.info(f"Loading data from vector store.{directory_path}")
    return Chroma(persist_directory=directory_path,
                  embedding_function=OpenAIEmbeddings())


def create_embedding_from_pdf(source_path: str, persisted_path: str) -> Chroma:
    """
    Reponsible for loading multiple pdf's and perform embeddings.
    :param source_path: Read PDF files.
    :param persisted_path: Saving embedding in path
    :return:
    """
    try:
        logger.info(f"Creating vector embeddings from PDF files from directory path.{source_path} and saving it in "
                    f"{persisted_path} for next time usage.")
        _create_dir_if_not_exists(source_path)
        _create_dir_if_not_exists(persisted_path)
        pdf_files = [os.path.join(source_path, f) for f in os.listdir(source_path) if f.endswith('.pdf')]
        documents = []
        for pdf_path in pdf_files:
            logger.debug(f"pdf_path{pdf_path}")
            loader = PyPDFLoader(pdf_path)
            documents.extend(loader.load())
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        splits = text_splitter.split_documents(documents)
        return Chroma.from_documents(documents=splits, embedding=OpenAIEmbeddings(),
                                     persist_directory=persisted_path)
    except Exception as e:
        logger.error(f"An issue has found. Manual intervention is required {e}")
        raise e


def _create_dir_if_not_exists(dir_path: str):
    """
    Create directory if does not exist.
    :param dir_path:
    :return:
    """
    if not dir_path.exists():
        logger.info(f"The source path {dir_path} does not exist. Creating directory.")
        dir_path.mkdir(parents=True, exist_ok=True)
