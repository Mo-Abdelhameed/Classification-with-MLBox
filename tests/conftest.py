import json
import os
import random
import string
import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from src.schema.data_schema import ClassificationSchema
from src.serve import create_app
from src.serve_utils import get_model_resources
from src.train import run_training


@pytest.fixture
def schema_dict():
    """Fixture to create a sample schema for testing"""
    valid_schema = {
        "title": "test dataset",
        "description": "test dataset",
        "modelCategory": "multiclass_classification",
        "schemaVersion": 1.0,
        "inputDataFormat": "CSV",
        "encoding": "utf-8",
        "id": {"name": "id", "description": "unique identifier."},
        "target": {
            "name": "target_field",
            "description": "some target desc.",
            "classes": ["0", "1", "2"],
        },
        "features": [
            {
                "name": "numeric_feature_1",
                "description": "some desc.",
                "dataType": "NUMERIC",
                "example": 50,
                "nullable": True,
            },
            {
                "name": "numeric_feature_2",
                "description": "some desc.",
                "dataType": "NUMERIC",
                "example": 0.5,
                "nullable": False,
            },
            {
                "name": "categorical_feature_1",
                "description": "some desc.",
                "dataType": "CATEGORICAL",
                "categories": ["A", "B", "C"],
                "nullable": True,
            },
            {
                "name": "categorical_feature_2",
                "description": "some desc.",
                "dataType": "CATEGORICAL",
                "categories": ["A", "B", "C", "D", "E"],
                "nullable": False,
            },
        ],
    }
    return valid_schema


@pytest.fixture
def schema_provider(schema_dict):
    """Fixture to create a sample schema for testing"""
    return ClassificationSchema(schema_dict)


@pytest.fixture
def config_dir_path():
    """Fixture to create a sample config_dir_path"""
    cur_dir = os.path.dirname(os.path.abspath(__file__))
    config_dir_path = os.path.join(cur_dir, "..", "src", "config")
    return config_dir_path


@pytest.fixture
def model_config(config_dir_path):
    """Fixture to create a sample model_config json"""
    model_config_file = os.path.join(config_dir_path, "model_config.json")
    with open(model_config_file, "r", encoding="utf-8") as file:
        model_config = json.load(file)
    return model_config


@pytest.fixture
def preprocessing_config(config_dir_path):
    """Fixture to create a preprocessing config"""
    preprocessing_config_file = os.path.join(config_dir_path, "preprocessing.json")
    with open(preprocessing_config_file, "r", encoding="utf-8") as file:
        pp_config = json.load(file)
    return pp_config


@pytest.fixture
def preprocessing_config_file_path(preprocessing_config, tmpdir):
    """Fixture to create and save a sample preprocessing_config json"""
    config_file_path = tmpdir.join("preprocessing.json")
    with open(config_file_path, "w") as file:
        json.dump(preprocessing_config, file)
    return str(config_file_path)


@pytest.fixture
def sample_data():
    """Fixture to create a larger sample DataFrame for testing"""
    np.random.seed(0)
    n = 100
    data = pd.DataFrame(
        {
            "id": range(1, n + 1),
            "numeric_feature_1": np.random.randint(1, 100, size=n),
            "numeric_feature_2": np.random.normal(0, 1, size=n),
            "categorical_feature_1": np.random.choice(["A", "B", "C"], size=n),
            "categorical_feature_2": np.random.choice(
                ["P", "Q", "R", "S", "T"], size=n
            ),
            "target_field": np.random.choice(["0", "1", "2"], size=n),
        }
    )
    return data


@pytest.fixture
def sample_train_data(sample_data):
    """Fixture to create a larger sample DataFrame for testing"""
    N_train = int(len(sample_data) * 0.8)
    return sample_data.head(N_train)


@pytest.fixture
def sample_test_data(sample_data):
    """Fixture to create a larger sample DataFrame for testing"""
    N_test = int(len(sample_data) * 0.2)
    return sample_data.tail(N_test)


@pytest.fixture
def train_data_file_name():
    return "train.csv"


@pytest.fixture
def train_dir(sample_train_data, tmpdir, train_data_file_name):
    """Fixture to create and save a sample DataFrame for testing"""
    train_data_dir = tmpdir.mkdir("train")
    train_data_file_path = train_data_dir.join(train_data_file_name)
    sample_train_data.to_csv(train_data_file_path, index=False)
    return str(train_data_dir)


@pytest.fixture
def test_data_file_name():
    return "test.csv"


@pytest.fixture
def test_dir(sample_test_data, tmpdir, test_data_file_name, schema_provider):
    """Fixture to create and save a sample DataFrame for testing"""
    test_data_dir = tmpdir.mkdir("test")
    test_data_file_path = test_data_dir.join(test_data_file_name)
    sample_test_data = sample_test_data.drop(columns=schema_provider.target)
    sample_test_data.to_csv(test_data_file_path, index=False)
    return str(test_data_dir)


@pytest.fixture
def input_schema_file_name():
    return "schema.json"


@pytest.fixture
def input_schema_dir(schema_dict, tmpdir, input_schema_file_name):
    """Fixture to create and save a sample schema for testing"""
    schema_dir = tmpdir.mkdir("input_schema")
    schema_file_path = schema_dir.join(input_schema_file_name)
    with open(schema_file_path, "w") as file:
        json.dump(schema_dict, file)
    return str(schema_dir)


@pytest.fixture
def model_config_file_path(model_config, tmpdir):
    """Fixture to create and save a sample model_config json"""
    config_file_path = tmpdir.join("model_config.json")
    with open(config_file_path, "w") as file:
        json.dump(model_config, file)
    return str(config_file_path)


@pytest.fixture
def predictions_df():
    """Fixture for creating a DataFrame representing predictions."""
    num_preds = 50
    # Create 5 random probabilities
    probabilities_A = [random.uniform(0, 0.3) for _ in range(num_preds)]
    probabilities_B = [random.uniform(0, 0.3) for _ in range(num_preds)]

    summ = np.array(probabilities_A) + np.array(probabilities_B)
    # Subtract each probability from 1 to create a complementary probability
    probabilities_C = [1 - p for p in summ]

    # Create a DataFrame with an 'id' column and three class probability
    # columns 'A', 'B' and 'C'
    df = pd.DataFrame(
        {
            "id": [
                "".join(
                    random.choices(string.ascii_lowercase + string.digits, k=num_preds)
                )
                for _ in range(num_preds)
            ],
            "0": probabilities_A,
            "1": probabilities_B,
            "2": probabilities_C,
        }
    )
    return df


@pytest.fixture
def test_resources_dir_path(tmpdir):
    """Define a fixture for the path to the test_resources directory."""
    tmpdir.mkdir("test_resources")
    test_resources_path = os.path.join(tmpdir, "test_resources")
    return test_resources_path


@pytest.fixture
def config_file_paths_dict(
    model_config_file_path,
):
    """Define a fixture for the paths to all config files."""
    return {
        "model_config_file_path": model_config_file_path,
    }


@pytest.fixture
def resources_paths_dict(test_resources_dir_path, model_config_file_path):
    """Define a fixture for the paths to the test model resources."""
    return {
        "saved_schema_dir_path": os.path.join(test_resources_dir_path, "schema"),
        "predictor_dir_path": os.path.join(test_resources_dir_path, "predictor"),
        "results_dir_path": os.path.join(test_resources_dir_path, "results"),
        "model_config_file_path": model_config_file_path,
        "predictions_file_path": os.path.join(
            test_resources_dir_path, "predictions.csv"
        ),
    }


@pytest.fixture
def sample_request_data(schema_dict):
    # Define a fixture for test request data
    sample_dict = {
        # made up id for this test
        schema_dict["id"]["name"]: "42",
    }
    for feature in schema_dict["features"]:
        if feature["dataType"] == "NUMERIC":
            sample_dict[feature["name"]] = feature["example"]
        elif feature["dataType"] == "CATEGORICAL":
            sample_dict[feature["name"]] = feature["categories"][0]
    return {"instances": [{**sample_dict}]}


@pytest.fixture
def sample_response_data(schema_dict):
    # Define a fixture for expected response
    return {
        "status": "success",
        "message": "",
        "timestamp": "...varies...",
        "requestId": "...varies...",
        "targetDescription": schema_dict["target"]["description"],
        "predictions": [
            {
                "sampleId": "42",
                "prediction": 999,  # we do not test for prediction value
            }
        ],
    }


@pytest.fixture
def app(
    input_schema_dir,
    train_dir,
    resources_paths_dict: dict,
):
    """
    Define a fixture for the test app.

    Args:
        input_schema_dir (str): Directory path to the input data schema.
        train_dir (str): Directory path to the training data.
        resources_paths_dict (dict): Dictionary containing the paths to the
            resources files such as trained models, encoders, and explainers.
    """

    # Create temporary paths for all outputs/artifacts
    saved_schema_dir_path = resources_paths_dict["saved_schema_dir_path"]
    predictor_dir_path = resources_paths_dict["predictor_dir_path"]
    model_config_file_path = resources_paths_dict["model_config_file_path"]
    results_path = resources_paths_dict["results_dir_path"]

    # Run the training process
    run_training(
        input_schema_dir=input_schema_dir,
        saved_schema_dir_path=saved_schema_dir_path,
        train_dir=train_dir,
        predictor_dir_path=predictor_dir_path,
        result_path=results_path
    )

    # create model resources dictionary
    model_resources = get_model_resources(
        saved_schema_dir_path=saved_schema_dir_path,
        predictor_dir_path=predictor_dir_path,
        model_config_file_path=model_config_file_path,
    )
    # create test app
    return TestClient(create_app(model_resources))
