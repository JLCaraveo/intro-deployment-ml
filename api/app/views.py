from .models import PredictionRequest
from .utils import get_model, transform_to_dataframe


model = get_model()

def get_prediction(request: PredictionRequest) -> float:
    """
    Funtion to make a prediction according to a input data
    **Args:**
    request: Input data to make  prediction
    **Returns:**
    maximun value between 0 and prediction, in case there is a mistake in the calculation, eg: negative probabilities.
    """
    data_to_predict = transform_to_dataframe(request)
    prediction = model.predict(data_to_predict)[0]

    return max(0, prediction)