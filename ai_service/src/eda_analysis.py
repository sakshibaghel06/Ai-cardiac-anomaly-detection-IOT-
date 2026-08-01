import logging
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier

logger = logging.getLogger(__name__)


class EDAReport:
    def __init__(self, output_dir: str | Path) -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def generate(self, df: pd.DataFrame, target_column: str) -> None:
        logger.info('Generating EDA report')
        summary_path = self.output_dir / 'eda_summary.csv'
        df.info()
        summary = pd.DataFrame({
            'column': df.columns,
            'dtype': df.dtypes.astype(str),
            'missing_values': df.isna().sum(),
            'unique_values': [df[col].nunique(dropna=False) for col in df.columns],
        })
        summary.to_csv(summary_path, index=False)

        missing_plot_path = self.output_dir / 'missing_values.png'
        plt.figure(figsize=(10, 4))
        sns.heatmap(df.isna(), cbar=False, cmap='viridis')
        plt.title('Missing Values Heatmap')
        plt.tight_layout()
        plt.savefig(missing_plot_path)
        plt.close()

        target_plot_path = self.output_dir / 'target_distribution.png'
        plt.figure(figsize=(6, 4))
        df[target_column].value_counts(normalize=True).plot(kind='bar', color=['#4C78A8', '#F58518'])
        plt.title('Target Distribution')
        plt.xlabel(target_column)
        plt.ylabel('Proportion')
        plt.tight_layout()
        plt.savefig(target_plot_path)
        plt.close()

        corr_path = self.output_dir / 'correlation_matrix.png'
        numeric_df = df.select_dtypes(include=['number'])
        if len(numeric_df.columns) > 1:
            plt.figure(figsize=(10, 8))
            corr = numeric_df.corr(numeric_only=True)
            sns.heatmap(corr, annot=False, cmap='coolwarm')
            plt.title('Correlation Matrix')
            plt.tight_layout()
            plt.savefig(corr_path)
            plt.close()

        hist_path = self.output_dir / 'histograms.png'
        plt.figure(figsize=(14, 10))
        numeric_df.hist(figsize=(14, 10), bins=20)
        plt.tight_layout()
        plt.savefig(hist_path)
        plt.close()

        box_path = self.output_dir / 'boxplots.png'
        plt.figure(figsize=(14, 8))
        numeric_df.boxplot()
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(box_path)
        plt.close()

        if target_column in df.columns:
            feature_cols = [col for col in df.columns if col != target_column]
            X = pd.get_dummies(df[feature_cols], drop_first=True)
            y = df[target_column]
            model = RandomForestClassifier(n_estimators=100, random_state=42)
            model.fit(X, y)
            importance = pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=False)
            importance_path = self.output_dir / 'feature_importance.png'
            plt.figure(figsize=(10, 6))
            importance.head(15).plot(kind='barh', color='#4C78A8')
            plt.title('Feature Importance (Top 15)')
            plt.tight_layout()
            plt.savefig(importance_path)
            plt.close()
