"""Project pipelines."""
# src/titanic_survival_prediction/pipeline_registry.py

from __future__ import annotations

from typing import Dict
from kedro.pipeline import Pipeline
from titanic_survival_prediction.pipelins import de, ds

def register_pipelines() -> Dict[str, Pipeline]:
    """Registra os pipelines do projeto."""
    data_engineering_pipeline = de.create_pipeline()
    data_science_pipeline = ds.create_pipeline()

    return{
        "de": data_engineering_pipeline,
        "ds": data_science_pipeline,
        "__default__": data_engineering_pipeline + data_science_pipeline,
    }
