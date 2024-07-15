import logging
import os

import fitz

logger = logging.getLogger(__name__)


def process_pdfs_in_directory(directory_path):
    """
    Processes PDF files in a directory and extracts text from each page.
    :param directory_path: Path to the directory containing PDF files
    :return: A list of dictionaries containing page information and text
    """
    try:
        pdf_files = [f for f in os.listdir(directory_path) if f.endswith('.pdf')]
        pages_and_texts = []
        for i, pdf_file in enumerate(pdf_files):
            logger.info(f"PDF file {pdf_file}")
            pdf_path = os.path.join(directory_path, pdf_file)
            doc_id = f"doc_{i}"  # Generate a unique ID for each document
            _extract_text_from_pdf(pdf_path, doc_id, pages_and_texts)
            print(f"Uploaded {pdf_file} as {doc_id}")

        return pages_and_texts
    except Exception as e:
        logger.error(f"An error occurred while processing the directory {directory_path}: {e}")
        raise e


def _extract_text_from_pdf(pdf_path, doc_id, pages_and_texts):
    """
    Extracts text from a PDF file and appends it to the pages_and_texts list.

    :param pdf_path: Path to the PDF file
    :param doc_id: Unique ID for the document
    :param pages_and_texts: List to store extracted page information and text
    :return: The updated pages_and_texts list
    """
    try:
        document = fitz.open(pdf_path)
        for page_num, page in enumerate(document):
            try:
                text = page.get_text()
                text = _text_formatter(text=text)
                pages_and_texts.append({"page_number": f"{doc_id} - {str(page_num)}",
                                        "page_char_count": len(text),
                                        "page_word_count": len(text.split(" ")),
                                        "page_setence_count_raw": len(text.split(". ")),
                                        "page_token_count": len(text) / 4,  # 1 token = ~4 characters
                                        "text": text})
            except Exception as e:
                logger.error(f"Failed to extract text from page {page_num} of PDF file {pdf_path}: {e}")
                continue
    except Exception as e:
        logger.error(f"Failed to open PDF file {pdf_path}: {e}")
        raise e

    return pages_and_texts


def _text_formatter(text: str) -> str:
    """
    Formats the extracted text by removing newlines and trimming whitespace.
    :param text: The raw extracted text
    :return: The cleaned and formatted text
    """
    try:
        cleaned_text = text.replace("\n", " ").strip()
        return cleaned_text
    except Exception as e:
        logger.error(f"Failed to format text: {e}")
        raise e
