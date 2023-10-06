import json
from unittest.mock import patch

import pandas as pd
import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def predictor_dir(tmpdir):
    return tmpdir.mkdir("predictor")


def test_ping(app):
    """Test the /ping endpoint."""
    response = app.get("/ping")
    assert response.status_code == 200
    assert response.json() == {"message": "Pong!"}


@patch("src.serve.transform_req_data_and_make_predictions")
def test_infer_endpoint(mock_transform_and_predict, app, sample_request_data):
    """
    Test the infer endpoint.

    The function creates a mock request and sets the expected return value of the
    mock_transform_and_predict function.
    It then sends a POST request to the "/infer" endpoint with the mock request data.
    The function asserts that the response status code is 200 and the JSON response
    matches the expected output.
    Additionally, it checks if the mock_transform_and_predict function was called with
    the correct arguments.

    Args:
        mock_transform_and_predict (MagicMock): A mock of the
            transform_req_data_and_make_predictions function.
        app (TestClient): The TestClient fastapi app

    """
    # Define what your mock should return
    mock_transform_and_predict.return_value = pd.DataFrame(), {
        "status": "success",
        "predictions": [],
    }
    response = app.post("/infer", data=json.dumps(sample_request_data))

    print(response.json())
    assert response.status_code == 200
    assert response.json() == {"status": "success", "predictions": []}
    # You can add more assertions to check if the function was called with the
    # correct arguments
    mock_transform_and_predict.assert_called()
