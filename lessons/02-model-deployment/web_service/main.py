from fastapi import FastAPI
from pydantic import BaseModel
from lib.preprocessing import encode_categorical_cols, load_preprocessor
from lib.models import get_model
from sklearn.feature_extraction import DictVectorizer
from typing import List
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from app_config import PATH_TO_PREPROCESSOR, PATH_TO_MODEL, CATEGORICAL_COLS
import logging
# import logging.config

logger = logging.getLogger(__name__)

app = FastAPI()
class InputData(BaseModel):
    PULocationID: int
    DOLocationID: int
    passenger_count: int

def run_inference(user_input: List[InputData], dv: DictVectorizer, model: BaseEstimator) -> np.ndarray:
    df = pd.DataFrame([x.dict() for x in user_input])
    df = encode_categorical_cols(df)
    dicts = df[CATEGORICAL_COLS].to_dict(orient="records")
    X = dv.transform(dicts)
    y = model.predict(X)
    logger.info(f"Predicted trip duration: {y}")
    return y

@app.get("/")
def read_root():
    return {"message": "Trip duration prediction!"}

@app.post("/predict_duration")
def predict_duration_route(payload: InputData):
    dv = load_preprocessor(PATH_TO_PREPROCESSOR)
    model = get_model(PATH_TO_MODEL)
    y = run_inference([payload], dv, model)
    return {"Hello, your trip duration prediction is: ": y[0]}

