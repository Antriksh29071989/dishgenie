import logging.config
from pathlib import Path

from src.container.opensource.embeddings import data_embedder, embedding_pipeline
from src.container.opensource.models import open_llm
from src.container.opensource.prompts import prompt_loader
from src.container.opensource.utils import df_utils
from src.container.shared import constants


logging.config.fileConfig(constants.LOG_CONF, disable_existing_loggers=False)
embedding_path = "output/vector_embeddings.csv"

if Path(embedding_path).exists():
    logging.info(f"loading embedded file fro path {embedding_path}.")
    text_with_chunk_df = df_utils.load_csv(embedding_path)
else:
    directory_path = constants.SOURCE_PATH
    logging.info(f"loading pdf files from {directory_path} and saving embedding file at {embedding_path}.")
    embedding_pipeline.create_embedding_from_pdf(directory_path=directory_path, persisted_path=embedding_path)
    text_with_chunk_df = df_utils.load_csv(embedding_path)

embeddings = df_utils.convert_embeddings(text_with_chunk_df)
model_id = "microsoft/Phi-3-mini-4k-instruct"
logging.info(f"loading model {model_id}...")
llm_model, tokenizer = open_llm.load_model(model_id, device_type="cpu")

# Open Source embedding model https://huggingface.co/sentence-transformers/all-mpnet-base-v2
embedding_model = "all-mpnet-base-v2"
query = "The drain pump is still running, even with the door open. Can you please help ?"
scores, indices = data_embedder.retrieve_relevant_resources(query=query, embeddings=embeddings,
                                                            embedding_model=embedding_model,
                                                            device_type="cpu")

pages_and_chunks = text_with_chunk_df.to_dict(orient="records")

context_items = [pages_and_chunks[i] for i in indices]
for i, item in enumerate(context_items):
    item["score"] = scores[i].cpu()
context = "- " + "\n- ".join([item["sentence_chunk"] for item in context_items])

base_prompt = prompt_loader.get_prompt(context, query)
output = open_llm.predict(base_prompt, llm_model, tokenizer)
logging.info(f"Assistant:{output[0]['generated_text']}")
