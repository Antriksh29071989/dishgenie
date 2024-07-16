from unittest.mock import MagicMock

from src.container.opensource.preprocessing.data_cleaner import data_preprocessing, convert_to_chunks


def test_convert_to_chunks(mocker):
    mock_splitter = MagicMock()
    mocker.patch('src.container.opensource.preprocessing.data_cleaner.RecursiveCharacterTextSplitter',
                 return_value=mock_splitter)

    mock_splitter.split_text = MagicMock(return_value=["chunk1", "chunk2"])

    pages_and_texts = [{"sentences": ["Dishgenie is AI assistant.", "This error is related to LG dishwasher."]}]
    result = convert_to_chunks(pages_and_texts)

    assert "sentence_chunk" in result[0]
    assert result[0]["sentence_chunk"] == ["chunk1", "chunk2"]
    assert result[0]["num_chunks"] == 2
