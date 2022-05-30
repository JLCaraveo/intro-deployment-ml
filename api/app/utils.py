from joblib import load
from sklearn.pipeline import Pipeline
from pydantic import BaseModel
from pandas import DataFrame
import os
from io import BytesIO


def get_model() -> Pipeline:
    """
    Function to load the model
    **Returns:**
    loaded model
    """
    model_path = os.environ.get('MODEL_PATH', 'model/model.pkl')
    with open(model_path, 'rb') as model_file:
        model = load(BytesIO(model_file.read()))

    return model


def transform_to_dataframe(class_model: BaseModel) -> DataFrame:
    """
    Funtion to transform a model to dataframe
    **Args:**
    class_model: model to transform to DataFrame
    **Returns:**
    model converted to data_frame
    """
    transition_dictionary = {key: [value] for key, value in class_model.dict().items()}
    data_frame = DataFrame(transition_dictionary)

    return data_frame