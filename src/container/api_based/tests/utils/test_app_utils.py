from src.container.api_based.utils.app_utils import has_embeddings_created


def test_has_embeddings_created(mocker):
    mocker.patch('os.path.exists', return_value=True)
    mocker.patch('os.path.isdir', return_value=True)
    mocker.patch('os.listdir', return_value=['file1.txt', 'file2.txt'])
    mocker.patch('os.path.isfile', return_value=True)

    directory_path = "../../../source"
    result = has_embeddings_created(directory_path)

    assert result is True


def test_has_embeddings_created_no_files(mocker):
    mocker.patch('os.path.exists', return_value=True)
    mocker.patch('os.path.isdir', return_value=True)
    mocker.patch('os.listdir', return_value=[])
    mocker.patch('os.path.isfile', return_value=False)

    directory_path = "../../../source"
    result = has_embeddings_created(directory_path)

    assert result is False
