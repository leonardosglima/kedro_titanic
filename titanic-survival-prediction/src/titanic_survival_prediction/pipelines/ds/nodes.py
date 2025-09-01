"""
This is a boilerplate pipeline 'ds'
generated using Kedro 1.0.0
"""
# src/titanic_survival_prediction/pipelines/ds/nodes.py

import logging
from typing import Dict, Any
import pandas as pd
import xgboost as xgb
from sklearn.metrics import accuracy_score, f1_score

def train_model(X_train: pd.DataFrame, y_train: pd.Series) -> xgb.XGBClassifier:
    """Nó para treinar o modelo XGBoost."""

    # O XGBoost tem seus próprios parâmetros, que podemos adicionar ao parameters.yml
    # Por enquanto, usaremos valores padrão razoáveis.
    xgb_classifier = xgb.XGBClassifier(
        objective="binary:logistic",
        eval_metric="logloss",
        use_label_encoder=False,
        n_estimators=100,
        learning_rate=0.1,
        max_depth=3,
        random_state=42
    )

    xgb_classifier.fit(X_train, y_train)
    return xgb_classifier

def evaluate_model(model: xgb.XGBClassifier, X_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, Any]:
    """Nó para avaliar o modelo treinado."""

    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    logger = logging.getLogger(__name__)
    logger.info(f"Acurácia do modelo: {accuracy:.3f}")
    logger.info(f"F1-Score do modelo: {f1:.3f}")

    return {"accuracy": accuracy, "f1_score": f1}