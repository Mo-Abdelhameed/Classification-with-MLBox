import os

import pytest

from config.paths import INPUT_SCHEMA_DIR
from src.schema.data_schema import (
    SCHEMA_FILE_NAME,
    ClassificationSchema,
    load_json_data_schema,
    load_saved_schema,
    save_schema,
)


@pytest.fixture
def input_schema_dir():
    return INPUT_SCHEMA_DIR


def test_init(schema_provider):
    """
    Test the initialization of ClassificationSchema class with a valid schema
    dictionary.

    Asserts that the properties of the schema object match the input schema dictionary.
    """
    schema = schema_provider
    assert schema.model_category == "multiclass_classification"
    assert schema.title == "test dataset"
    assert schema.description == "test dataset"
    assert schema.schema_version == 1.0
    assert schema.input_data_format == "CSV"
    assert schema.id == "id"
    assert schema.target == "target_field"
    assert schema.target_classes == ["0", "1", "2"]
    assert schema.numeric_features == ["numeric_feature_1", "numeric_feature_2"]
    assert schema.categorical_features == [
        "categorical_feature_1",
        "categorical_feature_2",
    ]
    assert schema.features == [
        "numeric_feature_1",
        "numeric_feature_2",
        "categorical_feature_1",
        "categorical_feature_2",
    ]
    assert schema.all_fields == [
        "id",
        "target_field",
        "numeric_feature_1",
        "numeric_feature_2",
        "categorical_feature_1",
        "categorical_feature_2",
    ]


def test_get_allowed_values_for_categorical_feature(schema_provider):
    """
    Test the method to get allowed values for a categorical feature.
    Asserts that the allowed values match the input schema dictionary.
    Also tests for a ValueError when trying to get allowed values for a non-existent
    feature.
    """

    # When
    allowed_values = schema_provider.get_allowed_values_for_categorical_feature(
        "categorical_feature_2"
    )

    # Then
    assert allowed_values == ["A", "B", "C", "D", "E"]

    # When / Then
    with pytest.raises(ValueError):
        schema_provider.get_allowed_values_for_categorical_feature("Invalid feature")


def test_get_example_value_for_numeric_feature(schema_provider):
    """
    Test the method to get an example value for a numeric feature.
    Asserts that the example value matches the input schema dictionary.
    Also tests for a ValueError when trying to get an example value for a non-existent
    feature.
    """

    schema = schema_provider

    # When
    example_value = schema.get_example_value_for_feature("numeric_feature_1")

    # Then
    assert example_value == 50

    # When / Then
    with pytest.raises(ValueError):
        schema.get_example_value_for_feature("Invalid feature")


def test_get_description_for_id_target_and_features(schema_dict):
    """
    Test the methods to get descriptions for the id, target, and features.
    Asserts that the descriptions match the input schema dictionary.
    Also tests for a ValueError when trying to get a description for a non-existent
    feature.
    """
    schema = ClassificationSchema(schema_dict)

    # When
    id_description = schema.id_description
    target_description = schema.target_description
    feature_1_description = schema.get_description_for_feature("numeric_feature_1")
    feature_2_description = schema.get_description_for_feature("numeric_feature_1")

    # Then
    assert id_description == "unique identifier."
    assert target_description == "some target desc."
    assert feature_1_description == "some desc."
    assert feature_2_description == "some desc."

    # When / Then
    with pytest.raises(ValueError):
        schema.get_description_for_feature("Invalid Feature")


def test_is_feature_nullable():
    """
    Test the method to check if a feature is nullable.
    Asserts that the nullable status matches the input schema dictionary.
    Also tests for a ValueError when trying to check the nullable status for a
    non-existent feature.
    """
    # Given
    schema_dict = {
        "title": "test dataset",
        "description": "test dataset",
        "modelCategory": "multiclass_classification",
        "schemaVersion": 1.0,
        "inputDataFormat": "CSV",
        "id": {"name": "id", "description": "unique identifier."},
        "target": {
            "name": "target_field",
            "description": "some target desc.",
            "classes": ["A", "B"],
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
                "categories": ["P", "Q", "R", "S", "T"],
                "nullable": False,
            },
        ],
    }
    schema = ClassificationSchema(schema_dict)

    # When
    is_nullable = schema.is_feature_nullable("numeric_feature_1")

    # Then
    assert is_nullable is True

    # When
    is_not_nullable = schema.is_feature_nullable("numeric_feature_2")

    # Then
    assert is_not_nullable is False

    # When / Then
    with pytest.raises(ValueError):
        schema.is_feature_nullable("Invalid feature")


def test_load_json_data_schema(input_schema_dir):
    """
    Test the method to load a schema from a JSON file.
    Asserts that the properties of the schema object match the input schema dictionary.
    """
    # Given input_schema_dir

    # When
    schema = load_json_data_schema(input_schema_dir)

    # Then
    assert isinstance(schema, ClassificationSchema)
    assert schema.model_category == "multiclass_classification"


def test_save_and_load_schema(tmpdir, schema_provider):
    # Save the schema using the save_schema function
    save_dir_path = str(tmpdir)
    save_schema(schema_provider, save_dir_path)

    # Check if file was saved correctly
    file_path = os.path.join(save_dir_path, SCHEMA_FILE_NAME)
    assert os.path.isfile(file_path)

    # Load the schema using the load_saved_schema function
    loaded_schema = load_saved_schema(save_dir_path)
    # Check if the loaded schema is an instance of ClassificationSchema
    assert isinstance(loaded_schema, ClassificationSchema)
    assert loaded_schema.model_category == "multiclass_classification"


def test_load_saved_schema_nonexistent_file(tmpdir):
    # Try to load the schema from a non-existent file
    save_dir_path = os.path.join(tmpdir, "non_existent")

    with pytest.raises(FileNotFoundError):
        returned = load_saved_schema(save_dir_path)
        print(returned)