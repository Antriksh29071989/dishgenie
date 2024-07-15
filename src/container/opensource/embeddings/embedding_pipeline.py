import logging

from src.container.opensource.embeddings.data_embedder import generate_embeddings, save_embeddings
from src.container.opensource.loader.pdf_data_loader import process_pdfs_in_directory
from src.container.opensource.preprocessing import data_cleaner
from src.container.opensource.preprocessing.data_cleaner import data_preprocessing

logger = logging.getLogger(__name__)


def create_embedding_from_pdf(directory_path: str, persisted_path: str):
    logger.info(f"loading pdf from path {directory_path}")
    pages_and_texts = process_pdfs_in_directory(directory_path)
    logger.info("preprocessing data...")
    pages_and_texts = data_preprocessing(pages_and_texts)
    logger.info("Convert to chunks..")
    pages_and_texts = data_cleaner.convert_to_chunks(pages_and_texts)
    logger.info("Generate embeddings..")
    vector_embeddings = generate_embeddings(pages_and_texts)
    logger.info("Save embeddings at path {persisted_path}")
    save_embeddings(vector_embeddings, persisted_path)
