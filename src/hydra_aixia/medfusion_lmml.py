from __future__ import annotations

import os
from collections import Counter
from typing import Iterable

import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler


def fuse_results(*results: Iterable[int]) -> list[int]:
    """Fuse classifier outputs by majority vote."""
    rows = zip(*results)
    return [Counter(row).most_common(1)[0][0] for row in rows]


def preprocess_tabular_data(df: pd.DataFrame) -> pd.DataFrame:
    data = df.copy()
    data.ffill(inplace=True)
    for col in data.select_dtypes(include="object").columns:
        data[col] = LabelEncoder().fit_transform(data[col].astype(str))
    numeric = data.select_dtypes(include="number").columns
    data[numeric] = StandardScaler().fit_transform(data[numeric])
    return data


def train_random_forest(df: pd.DataFrame, target_column: str):
    X = preprocess_tabular_data(df.drop(columns=[target_column]))
    y = df[target_column]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    return {
        "model": model,
        "accuracy": accuracy_score(y_test, preds),
        "report": classification_report(y_test, preds),
    }


def summarize_with_openai(prompt: str, model: str = "gpt-4o-mini") -> str:
    """Summarize text using OpenAI. Requires OPENAI_API_KEY in the environment."""
    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is not set. Create a local .env file or export it in your shell.")
    client = OpenAI(api_key=api_key)
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "Summarize clearly, structurally, and concisely."},
            {"role": "user", "content": prompt},
        ],
    )
    return response.choices[0].message.content or ""
