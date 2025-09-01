"""
This is a boilerplate pipeline 'ds'
generated using Kedro 1.0.0
"""
# src/titanic_survival_prediction/pipelines/ds/pipeline.py


from kedro.pipeline import node, Pipeline  # noqa
from kedro.pipeline.modular_pipeline import pipeline
from .nodes import train_model, evaluate_model

def create_pipeline(**kwargs) -> Pipeline:
    return Pipeline([
        node(
            func=train_model,
            inputs=['X_train', 'y_train'],
            outputs='classifier',
            name='train_model_node',
        ),
        node(
            func=evaluate_mode,
            inputs=['classifier', 'X_test', 'y_test'],
            outputs='metrics',
            name='evaluate_model_node',
        ),
    ]
)