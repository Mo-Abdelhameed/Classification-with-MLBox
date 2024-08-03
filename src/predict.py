import os
import pandas as pd
from config import paths
from logger import get_logger
from Classifier import Classifier, predict_with_model
from schema.data_schema import load_saved_schema
from utils import read_csv_in_directory, save_dataframe_as_csv, ResourceTracker

logger = get_logger(task_name="predict")


def run_batch_predictions(
    test_dir=paths.TEST_DIR,
    predictor_dir=paths.PREDICTOR_DIR_PATH,
    predictions_file_path=paths.PREDICTIONS_FILE_PATH,
    saved_schema_dir=paths.SAVED_SCHEMA_DIR_PATH,
    return_proba: bool = True
) -> None:
    """
    Run batch predictions on test data, save the predicted probabilities to a CSV file.

    This function reads test data from the specified directory,
    loads the preprocessing pipeline and pre-trained predictor model,
    transforms the test data using the pipeline,
    makes predictions using the trained predictor model,
    adds ids into the predictions dataframe,
    and saves the predictions as a CSV file.
    """
    with ResourceTracker(logger, monitoring_interval=0.1):
        x_test = read_csv_in_directory(test_dir)
        data_schema = load_saved_schema(saved_schema_dir)
        ids = x_test[data_schema.id]
        x_test.drop(columns=data_schema.id, inplace=True)

        for column in data_schema.categorical_features:
            x_test[column] = x_test[column].astype(str)

        model = Classifier.load(predictor_dir)

        logger.info("Making predictions...")
        predict_with_model(model, x_test)

    prediction_file_name = f"{data_schema.target}_predictions.csv"
    prediction_file_path = os.path.join(model.result_path, prediction_file_name)
    predictions_df = pd.read_csv(prediction_file_path)

    if return_proba:
        predictions_df = predictions_df[data_schema.target_classes]
    else:
        predictions_df = predictions_df[[f"{data_schema.target}_predicted"]]

    predictions_df[data_schema.id] = ids

    logger.info("Saving predictions...")
    save_dataframe_as_csv(dataframe=predictions_df, file_path=predictions_file_path)
    logger.info("Batch predictions completed successfully")


if __name__ == "__main__":
    run_batch_predictions()
