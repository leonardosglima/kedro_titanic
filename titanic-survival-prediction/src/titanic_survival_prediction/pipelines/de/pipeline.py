"""
This is a boilerplate pipeline 'de'
generated using Kedro 1.0.0
"""

# src/titanic_survival_prediction/pipelines/de/pipeline.py

from kedro.pipeline import Pipeline, node
from kedro.pipeline.modular_pipeline import pipeline
from .nodes import preprocess_titanic_data, split_data

def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                func=preprocess_titanic_data,
                inputs="titanic_raw_data",
                outputs="preprocess_titanic_node",
            ),
            node(
                func=split_data,
                inputs=["preprocessed_titanic", "params:data_science"],
                outputs=["X_train", "X_test", "y_train", "y_test"],
                name="split_data_node",
            )
        ]
    )