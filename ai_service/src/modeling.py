import logging
import warnings
from pathlib import Path
from typing import Any, Dict, List, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, StratifiedKFold, cross_val_predict
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

warnings.filterwarnings("ignore")
logger = logging.getLogger(__name__)


class HeartDiseaseModelTrainer:
    def __init__(self, output_dir: str | Path = "models") -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.models: Dict[str, Any] = {}
        self.best_model_name: str | None = None
        self.best_model: Any = None
        self.best_params: Dict[str, Any] | None = None
        self.results: List[Dict[str, Any]] = []

    def build_models(self) -> Dict[str, Any]:
        models = {
            "Logistic Regression": LogisticRegression(max_iter=5000, random_state=42),
            "Decision Tree": DecisionTreeClassifier(random_state=42),
            "Random Forest": RandomForestClassifier(random_state=42, n_estimators=200),
            "XGBoost": XGBClassifier(eval_metric="logloss", random_state=42, n_estimators=200, use_label_encoder=False),
            "LightGBM": LGBMClassifier(random_state=42, n_estimators=200),
            "CatBoost": CatBoostClassifier(verbose=False, random_state=42, iterations=200),
            "SVM": SVC(probability=True, random_state=42),
            "KNN": KNeighborsClassifier(n_neighbors=5),
        }
        self.models = models
        return models

    def tune_model(self, model_name: str, model: Any, X: pd.DataFrame, y: pd.Series) -> Any:
        param_grids = {
            "Logistic Regression": {
                "C": [0.1, 1, 10],
                "solver": ["liblinear", "lbfgs"],
            },
            "Decision Tree": {
                "max_depth": [3, 5, 7, None],
                "min_samples_split": [2, 5, 10],
                "criterion": ["gini", "entropy"],
            },
            "Random Forest": {
                "n_estimators": [100, 200],
                "max_depth": [3, 5, None],
                "min_samples_leaf": [1, 2, 4],
            },
            "XGBoost": {
                "n_estimators": [100, 200],
                "max_depth": [3, 5],
                "learning_rate": [0.05, 0.1],
            },
            "LightGBM": {
                "n_estimators": [100, 200],
                "max_depth": [3, 5],
                "learning_rate": [0.05, 0.1],
            },
            "CatBoost": {
                "iterations": [100, 200],
                "depth": [3, 6],
                "learning_rate": [0.05, 0.1],
            },
            "SVM": {
                "C": [0.1, 1, 10],
                "kernel": ["linear", "rbf"],
            },
            "KNN": {
                "n_neighbors": [3, 5, 7],
                "weights": ["uniform", "distance"],
            },
        }

        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        search = RandomizedSearchCV(
            estimator=model,
            param_distributions=param_grids[model_name],
            n_iter=10,
            cv=cv,
            scoring="roc_auc",
            n_jobs=-1,
            random_state=42,
        )
        search.fit(X, y)
        return search.best_estimator_, search.best_params_

    def evaluate_model(self, model: Any, X_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, Any]:
        preds = model.predict(X_test)
        probas = model.predict_proba(X_test)[:, 1]
        report = {
            "accuracy": round(accuracy_score(y_test, preds), 4),
            "precision": round(precision_score(y_test, preds), 4),
            "recall": round(recall_score(y_test, preds), 4),
            "f1": round(f1_score(y_test, preds), 4),
            "roc_auc": round(roc_auc_score(y_test, probas), 4),
            "confusion_matrix": confusion_matrix(y_test, preds).tolist(),
        }
        return report

    def train_and_compare(self, X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
        base_models = self.build_models()
        X_train, X_test, y_train, y_test = self._split_data(X, y)
        results = []

        for name, model in base_models.items():
            logger.info("Tuning %s", name)
            tuned_model, params = self.tune_model(name, model, X_train, y_train)
            tuned_model.fit(X_train, y_train)
            metrics = self.evaluate_model(tuned_model, X_test, y_test)
            metrics.update({"model": name, "params": params})
            results.append(metrics)
            self.models[name] = tuned_model
            self.best_params = params

        self.results = results
        results_df = pd.DataFrame(results)
        best_row = results_df.sort_values(by=["roc_auc", "f1"], ascending=False).iloc[0]
        self.best_model_name = best_row["model"]
        self.best_model = self.models[self.best_model_name]
        self.save_results(results_df)
        return results_df

    def _split_data(self, X: pd.DataFrame, y: pd.Series) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        from sklearn.model_selection import train_test_split

        return train_test_split(X, y, test_size=0.25, stratify=y, random_state=42)

    def save_results(self, results_df: pd.DataFrame) -> None:
        results_df.to_csv(self.output_dir / "model_comparison.csv", index=False)

    def save_artifacts(self, preprocessor: Any) -> None:
        joblib.dump(self.best_model, self.output_dir / "best_model.joblib")
        joblib.dump(preprocessor, self.output_dir / "preprocessing_pipeline.joblib")

    def load_artifacts(self) -> Tuple[Any, Any]:
        best_model = joblib.load(self.output_dir / "best_model.joblib")
        preprocessor = joblib.load(self.output_dir / "preprocessing_pipeline.joblib")
        return best_model, preprocessor
