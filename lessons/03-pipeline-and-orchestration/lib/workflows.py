import os
from typing import Optional
from loguru import logger
from lib.preprocessing import process_data
from lib.modeling import train_model, predict, evaluate_model
from lib.helpers import save_pickle, load_pickle
from sklearn.linear_model import LinearRegression
from sklearn.feature_extraction import DictVectorizer
import numpy as np
from prefect import flow

@flow
def train_model_workflow(
    train_filepath: str,
    test_filepath: str,
    artifacts_filepath: Optional[str] = None,
) -> dict:
    # Process data
    logger.debug(f"Processing train data from {train_filepath}")
    X_train, y_train, dv_train = process_data(train_filepath, with_target=True)

    logger.debug(f"Processing test data from {test_filepath}")
    X_test, y_test, dv_test = process_data(test_filepath, dv=dv_train, with_target=True)

    # Train model
    model = train_model(X_train, y_train)

    # Evaluate model
    y_pred = predict(X_test)
    score = evaluate_model(y_test, y_pred)

    # Save artifacts
    logger.debug(f"Saving artifacts to {artifacts_filepath}")
    save_pickle(artifacts_filepath, {'model': model, 'dv': dv_train})

    return {'train_score': score}

@flow
def batch_predict_workflow(
    input_filepath: str,
    model: Optional[LinearRegression] = None,
    dv: Optional[DictVectorizer] = None,
    artifacts_filepath: Optional[str] = None,
) -> np.ndarray:
    logger.debug(f"Loading artifacts from {artifacts_filepath}")
    artifacts = load_pickle(artifacts_filepath)
    model = artifacts['model']
    dv = artifacts['dv']

    # Process data without target column
    logger.debug(f"Processing input data from {input_filepath}")
    X, _, _ = process_data(input_filepath, dv=dv, with_target=False)

    # Predict
    y_pred = model.predict(X)
    return y_pred
