import logging
from pathlib import Path
from typing import Any, Dict, Tuple

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

logger = logging.getLogger(__name__)


class DataProcessor:
    def __init__(self, target_column: str = "HeartDisease") -> None:
        self.target_column = target_column
        self.preprocessor: ColumnTransformer | None = None
        self.feature_columns: list[str] = []

    def load_and_select_dataset(self, dataset_dir: str | Path) -> Tuple[pd.DataFrame, str]:
        dataset_dir = Path(dataset_dir)
        datasets = {
            "heart.csv": pd.read_csv(dataset_dir / "heart.csv"),
            "heart_cleveland_upload.csv": pd.read_csv(dataset_dir / "heart_cleveland_upload.csv"),
        }

        for name, df in datasets.items():
            logger.info("Loaded %s with shape %s", name, df.shape)

        df_a = datasets["heart.csv"].copy()
        df_b = datasets["heart_cleveland_upload.csv"].copy()

        if self._can_merge(df_a, df_b):
            logger.info("Merging both datasets after schema alignment")
            merged = self._merge_datasets(df_a, df_b)
            return merged, "merged"

        if len(df_a) >= len(df_b):
            logger.info("Using heart.csv as the primary dataset")
            return df_a, "heart.csv"

        logger.info("Using heart_cleveland_upload.csv as the primary dataset")
        return df_b, "heart_cleveland_upload.csv"

    def _can_merge(self, df_a: pd.DataFrame, df_b: pd.DataFrame) -> bool:
        mapping = {
            "Age": "age",
            "Sex": "sex",
            "ChestPainType": "cp",
            "RestingBP": "trestbps",
            "Cholesterol": "chol",
            "FastingBS": "fbs",
            "RestingECG": "restecg",
            "MaxHR": "thalach",
            "ExerciseAngina": "exang",
            "Oldpeak": "oldpeak",
            "ST_Slope": "slope",
            "HeartDisease": "condition",
        }
        return set(mapping.keys()) <= set(df_a.columns) and set(mapping.values()) <= set(df_b.columns)

    def _merge_datasets(self, df_a: pd.DataFrame, df_b: pd.DataFrame) -> pd.DataFrame:
        df_a = df_a.rename(columns={
            "Age": "age",
            "Sex": "sex",
            "ChestPainType": "cp",
            "RestingBP": "trestbps",
            "Cholesterol": "chol",
            "FastingBS": "fbs",
            "RestingECG": "restecg",
            "MaxHR": "thalach",
            "ExerciseAngina": "exang",
            "Oldpeak": "oldpeak",
            "ST_Slope": "slope",
            "HeartDisease": "condition",
        })

        df_b = df_b.rename(columns={
            "age": "age",
            "sex": "sex",
            "cp": "cp",
            "trestbps": "trestbps",
            "chol": "chol",
            "fbs": "fbs",
            "restecg": "restecg",
            "thalach": "thalach",
            "exang": "exang",
            "oldpeak": "oldpeak",
            "slope": "slope",
            "condition": "condition",
        })

        for col in ["sex", "cp", "restecg", "exang", "slope"]:
            df_a[col] = df_a[col].astype(str).str.strip().str.lower()
            df_b[col] = df_b[col].astype(str).str.strip().str.lower()

        df_a["sex"] = df_a["sex"].replace({"m": "male", "f": "female"})
        df_b["sex"] = df_b["sex"].replace({"1": "male", "0": "female"})

        df_a["cp"] = df_a["cp"].replace({"ata": "typical_angina", "nap": "atypical_angina", "asp": "non_anginal", "ta": "non_anginal"})
        df_b["cp"] = df_b["cp"].replace({0: "typical_angina", 1: "atypical_angina", 2: "non_anginal", 3: "asymptomatic"})

        df_a["restecg"] = df_a["restecg"].replace({"normal": "normal", "st": "st_t_wave_abnormality", "lvh": "left_ventricular_hypertrophy"})
        df_b["restecg"] = df_b["restecg"].replace({0: "normal", 1: "st_t_wave_abnormality", 2: "left_ventricular_hypertrophy"})

        df_a["exang"] = df_a["exang"].replace({"y": "yes", "n": "no"})
        df_b["exang"] = df_b["exang"].replace({0: "no", 1: "yes"})

        df_a["slope"] = df_a["slope"].replace({"up": "upsloping", "flat": "flat", "down": "downsloping"})
        df_b["slope"] = df_b["slope"].replace({0: "upsloping", 1: "flat", 2: "downsloping"})

        merged = pd.concat([df_a, df_b], ignore_index=True)
        merged = merged.drop_duplicates().reset_index(drop=True)

        common_columns = ["age", "sex", "cp", "trestbps", "chol", "fbs", "restecg", "thalach", "exang", "oldpeak", "slope", "condition"]
        merged = merged[[col for col in common_columns if col in merged.columns]].copy()
        return merged

    def build_preprocessor(self, X: pd.DataFrame) -> ColumnTransformer:
        numeric_features = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
        categorical_features = X.select_dtypes(exclude=["int64", "float64"]).columns.tolist()

        numeric_transformer = Pipeline(steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ])
        categorical_transformer = Pipeline(steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ])

        self.preprocessor = ColumnTransformer(
            transformers=[
                ("num", numeric_transformer, numeric_features),
                ("cat", categorical_transformer, categorical_features),
            ]
        )
        self.feature_columns = list(X.columns)
        return self.preprocessor

    def prepare_features(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        X = df.drop(columns=[self.target_column])
        y = df[self.target_column]
        self.build_preprocessor(X)
        return X, y

    def transform_features(self, X: pd.DataFrame) -> pd.DataFrame:
        if self.preprocessor is None:
            raise ValueError("Preprocessor has not been built yet")
        transformed = self.preprocessor.fit_transform(X)
        return pd.DataFrame(transformed)
