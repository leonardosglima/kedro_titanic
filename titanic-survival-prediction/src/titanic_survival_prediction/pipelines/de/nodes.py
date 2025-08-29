"""
This is a boilerplate pipeline 'de'
generated using Kedro 1.0.0
"""
# src/titanic_survival_prediction/pipelines/de/nodes.py

import pandas as pd
from sklearn.model_selection import train_test_split
from typing import Dict, List, Tuple

def _create_family_size(df: pd.DataFrame) -> pd.DataFrame:
    """Cria a feature 'FamilySize' somando 'SibSp' e 'Parch'."""
    df['FamilySize'] = df['SibSp'] + df['Parch'] + 1
    return df

def _extract_title(df: pd.DataFrame) -> pd.DataFrame:
    """Extrai o título do passageiro a partir da coluna 'Name'."""
    df['Title'] = df['Name'].str.extract(' ([A-Za-z]+)\.', expand=False)
    # Agrupa títulos raros em uma única categoria 'Rare'
    rare_titles = (df['Title'].value_counts() < 10)
    df['Title'] = df['Title'].apply(lambda x: 'Rare' if rare_titles[x] else x)
    return df

def preprocess_titanic_data(df: pd.DataFrame) -> pd.DataFrame:
    """Nó para pré-processar os dados do Titanic. Inclui tratamento de valores faltantes e engenharia de features."""
    # 1. Engenharia de Features
    df = _create_family_size(df)
    df = _extract_title(df)

    # 2. Tratamento de valores faltantes (Imputação)
    # Preenche a idade com a mediana, agrupada por sexo e classe
    df['Age'] = df.groupby(['Sex', 'Pclass']).transform(lambda x: x.fillna(x.median()))
    # Preenche a porta de embarque com a mais comum (moda)
    df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode()[0])

    # 3. Mapeamento de variáveis categóricas para numéricas
    title_maping = {"Mr": 1,
                    "Miss": 2,
                    "Mrs": 3,
                    "Master": 4,
                    "Rare": 5}
    df['Title'] = df['Title'].map(title_mappiing)
    df['Sex'] =df['Sex'].map({'male': 0, 'female': 1})
    df['Embarked'] = df['Embarked'].map({'S': 0, 'C': 1, 'Q': 2})

    # 4. Seleção e remoção de colunas não utilizadas
    df = df.drop(['PassengerId', 'Name', 'Ticket', 'Cabin', 'SibSp', 'Parch'], axis=1)

    # Garante que não haja mais NaNs (exceto em 'Survived', a variável alvo)
    df = df.dropna(subset=df.columns.difference(['Survived']))

    return df

def split_data(data: pd.DataFrame, parameters: Dict) -> Tuple:
    """Nó para dividir os dados em conjuntos de treino e teste."""
    X = data.drop(columns=parameters["target_column"])
    y = data[parameters["target_column"]]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=parameters["test_size"],
        random_state=parameters["random_state"],
    )
    return X_train, X_test, y_train, y_test
