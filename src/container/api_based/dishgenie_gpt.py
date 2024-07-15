from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory

from src.container.api_based.embeddings import vector_embedder
from src.container.api_based.models import gpt
from src.container.api_based.prompts import prompt_loader
from src.container.api_based.utils import app_utils, constants
import logging.config
from langchain_chroma import Chroma

logging.config.fileConfig(constants.LOG_CONF, disable_existing_loggers=False)

gpt_key = constants.GPT_KEY

logging.info("Setting up GPT key...")
app_utils.set_gpt_key(key=gpt_key)
persisted_path = constants.DB_PATH

logging.info("Validate if embedding has been created else create embeddings from source.")

if app_utils.has_embeddings_created(persisted_path):
    logging.debug("Existing embeddings found.")
    vector_store: Chroma = vector_embedder.load_data_from_vector_store(directory_path=persisted_path)
else:
    logging.debug("Creating new embeddings from source.")
    source_path = constants.SOURCE_PATH
    vector_store: Chroma = vector_embedder.create_embedding_from_pdf(source_path, persisted_path)

retriever = vector_store.as_retriever()
context_system_prompt = prompt_loader.contextualize_system_prompt

model_name = "gpt-3.5-turbo"
temperature = 0
logging.info(f"loading GPT model...{model_name} and {temperature}")
llm = gpt.load_model(model_name=model_name, temperature=temperature)

contextualize_q_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", context_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)
history_aware_retriever = create_history_aware_retriever(
    llm, retriever, contextualize_q_prompt
)

system_prompt = prompt_loader.system_prompt
qa_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)
question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)
store = {}


def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]


conversational_rag_chain = RunnableWithMessageHistory(
    rag_chain,
    get_session_history,
    input_messages_key="input",
    history_messages_key="chat_history",
    output_messages_key="answer",
)
logging.info("DishGenie is ready to provide answers, Please ask ...")

# Testing
questions = [
    "I ’m having trouble with a Model 18 ADA dishwasher. It’s showing an error code E4 and the customer is "
    "complaining is it not draining.",
    "Yes, I’ve checked it and there doesn’t seem to be any physical obstruction."
    "I have accessed the pump. There’s some debris here. I’ll clean it out and see if that fixes the issue.",
    "It works!"
]
for question in questions:
    logging.info(f"Technician:{question}")
    response = conversational_rag_chain.invoke(
        {"input": question},
        config={
            "configurable": {"session_id": "default"}
        })
    logging.info(f"DishGenie:{response['answer']}")
