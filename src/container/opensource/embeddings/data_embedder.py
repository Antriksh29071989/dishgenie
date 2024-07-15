import logging

import pandas as pd
import torch
from sentence_transformers import util, SentenceTransformer
logger = logging.getLogger(__name__)


def generate_embeddings(pages_and_chunks):
    """ Generate embeddings for the lodaed documents
    :param pages_and_chunks: A list of dictionaries containing page numbers and sentence chunks
    :return: A list of dictionaries with embedding information
    """
    try:
        embedding_model = SentenceTransformer(model_name_or_path="all-mpnet-base-v2",
                                              device="cpu")

        vector_embeddings = []
        logger.info("creating embeddings..")
        for sentences in pages_and_chunks:
            for item in sentences["sentence_chunk"]:
                if len(item) > 5:
                    chunk_dict = {"page_number": sentences["page_number"], "chunk_char_count": len(item),
                                  "chunk_word_count": len([word for word in item.split(" ")]),
                                  "chunk_token_count": len(item) / 4, "sentence_chunk": item,
                                  "embeddings": embedding_model.encode(item)}

                    vector_embeddings.append(chunk_dict)
                    return vector_embeddings

    except Exception as e:
        logger.error(f"Failed to generate embeddings for chunk {e}")
        raise e


def save_embeddings(vector_embeddings, file_path):
    """
    Save embeddings on local in a file.
    :param vector_embeddings: List of dictionaries containing embeddings and metadata
    :param file_path: Path to save the embeddings CSV file
    :return: None
    """
    try:
        text_chunks_and_embeddings_df = pd.DataFrame(vector_embeddings)
        logger.info(f"creating embeddings at path {file_path}")
        text_chunks_and_embeddings_df.to_csv(file_path, index=False)
    except Exception as e:
        logger.error(f"Failed to save embeddings to {file_path}: {e}")
        raise


def retrieve_relevant_resources(query: str,
                                embeddings: torch.tensor, embedding_model: str,
                                n_resources_to_return: int = 5, device_type: str = "cpu"):
    """
    Embeds a query with model and returns top k scores and indices from embeddings.
    :param query: Query string to find relevant resources for
    :param embeddings: Tensor of embeddings to compare the query against
    :param embedding_model: Path or name of the embedding model to use
    :param device_type: Device type to run the model on ('cpu' or 'cuda')
    :param n_resources_to_return: Number of top resources to return
    :return: Tuple of scores and indices of the relevant resources
    """


    try:
        logger.info("Finding relevant resources...")
        embedding_model = SentenceTransformer(model_name_or_path=embedding_model).to(device_type)
        query_embedding = embedding_model.encode(query, convert_to_tensor=True).to(device_type)
        query_embedding = query_embedding.to(torch.float32)
        dot_scores = util.dot_score(query_embedding, embeddings)[0].to(device_type)
        scores, indices = torch.topk(input=dot_scores,
                                     k=n_resources_to_return)
        return scores, indices
    except Exception as e:
        logger.error(f"Failed to retrieve relevant resources: {e}")
        raise e
