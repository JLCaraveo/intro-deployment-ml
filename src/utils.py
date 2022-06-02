from joblib import dump
from sklearn.pipeline import Pipeline
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def update_model(model: Pipeline) -> None:
    """
    Funtion to save a model in pkl format
    **Args:*
    model: Model to be saved
    """
    dump(model, 'model/model.pkl')


def save_simple_metrics_report(train_score: float, test_score: float, validation_score: float, model: Pipeline) -> None:
    """
    Funtion to save the metrics report.
    **Args:**
    train_score: model train score (float)
    test_score: model test score (float)
    validation_score: model validation score (float)
    model: Working model
    """
    with open('report.txt', 'w') as report_file:
        report_file.write('# Model Pipeline description\n')

        for key, value in model.named_steps.items():
            report_file.write(f'### {key}:{value.__repr__()}\n')

        report_file.write(f'## Train Score: {train_score}\n')
        report_file.write(f'## Test Score: {test_score}\n')
        report_file.write(f'## Validation Score: {validation_score}\n')


def get_model_performance_test_set(y_true: pd.Series, y_pred: pd.Series) -> None:
    """
    Funtion to plot the behavior model prediction and saved as png image
    **Args:**
    y_true: Real values
    y_pred: Predicted values
    """
    fig, ax = plt.subplots()
    fig.set_figheight(8)
    fig.set_figwidth(8)
    sns.regplot(x=y_pred, y=y_true)
    ax.set_xlabel('Predicted worldwide gross')
    ax.set_ylabel('Real worldwide gross')
    ax.set_title('Behavior of model prediction')
    fig.savefig('prediction_behavior.png')