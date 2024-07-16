import logging.config

from langchain_core.messages import HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.chat_history import BaseChatMessageHistory, InMemoryChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory

from src.container.scheduler.models import model_loader
from src.container.scheduler.utils import app_utils
from src.container.shared import constants

logging.config.fileConfig(constants.LOG_CONF, disable_existing_loggers=False)

gpt_key = constants.GPT_KEY

logging.info("Setting up GPT key...")
app_utils.set_gpt_key(key=gpt_key)

model_name = "gpt-3.5-turbo"
temperature = 0
model = model_loader.load_model(model_name, temperature)
prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a helpful dummy appointment assistant. "
            "You confirm the user in text message about booking the appointment",
        ),
        MessagesPlaceholder(variable_name="messages"),
    ]
)
store = {}


def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in store:
        store[session_id] = InMemoryChatMessageHistory()
    return store[session_id]


chain = prompt | model
with_message_history = RunnableWithMessageHistory(chain, get_session_history, input_messages_key="messages")

config = {"configurable": {"session_id": "default"}}
response = with_message_history.invoke(
    {"messages": [HumanMessage(content="Can you help me in booking an appointment?")], "language": "English"},
    config=config,
)
print(response.content)
response = with_message_history.invoke(
    {"messages": [
        HumanMessage(content="Book an appointment on date 12th July, time 12:00 and reason fixing dihwasher")],
        "language": "English"},
    config=config,
)
print(response.content)
