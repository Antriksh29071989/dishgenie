import logging

import numpy as np
import pandas as pd
import torch
from pandas import DataFrame

logger = logging.getLogger(__name__)


def load_csv(file_path) -> DataFrame:
    """
    Load a CSV file into a DataFrame.
    :param file_path: Path to the CSV file to load.
    :return: DataFrame containing the loaded CSV data.
    """
    try:
        logger.info(f"Loading file from path {file_path}")
        return pd.read_csv(file_path)
    except FileNotFoundError as e:
        logger.error(f"File not found: {file_path}")
        raise e
    except Exception as e:
        logger.error(f"An error occurred while loading the CSV file at {file_path}: {e}")
        raise e


def convert_embeddings(text_with_chunk_df):
    """
    Convert text embeddings from a DataFrame column into a tensor.

    :param text_with_chunk_df: DataFrame with a column 'embeddings' containing string representations of embeddings.
    :return: Tensor of embeddings.
    """
    try:
        text_with_chunk_df["embeddings"] = text_with_chunk_df["embeddings"].apply(
            lambda x: np.fromstring(x.strip("[]"), sep=" "))
        return torch.tensor(np.stack(text_with_chunk_df["embeddings"].tolist(), axis=0),
                            dtype=torch.float32)
    except Exception as e:
        logger.error(f"An error occurred while converting embeddings: {e}")
        raise e
