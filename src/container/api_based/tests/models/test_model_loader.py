# Assuming the code above is in a file named vector_store.py
# from model_loader import load_model
from src.container.api_based.models.model_loader import load_model


def test_load_model(mocker):
    mock_chat_openai = mocker.patch('src.container.api_based.models.model_loader.ChatOpenAI')
    model_name = "gpt-3.5-turbo"
    temperature = 0.7
    result = load_model(model_name, temperature)
    mock_chat_openai.assert_called_once_with(model=model_name, temperature=temperature)
    assert result == mock_chat_openai()
