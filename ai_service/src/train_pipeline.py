import logging
from pathlib import Path

import pandas as pd

from src.data_processing import DataProcessor
from src.modeling import HeartDiseaseModelTrainer

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def main() -> None:
    base_dir = Path(__file__).resolve().parent.parent
    dataset_dir = base_dir / 'dataset'

    processor = DataProcessor(target_column='condition')
    df, source = processor.load_and_select_dataset(dataset_dir)
    logger.info('Selected dataset source: %s', source)

    df = df.copy()
    if 'condition' not in df.columns:
        raise ValueError('Target column condition not found after preprocessing')

    if df['condition'].dtype == 'object':
        df['condition'] = df['condition'].astype(str).str.strip().str.lower()
        df['condition'] = df['condition'].replace({'yes': 1, 'no': 0, 'true': 1, 'false': 0, '1': 1, '0': 0})
    df['condition'] = pd.to_numeric(df['condition'], errors='coerce').astype('Int64')
    df = df.dropna(subset=['condition'])
    df['condition'] = df['condition'].astype(int)

    from src.eda_analysis import EDAReport

    eda_report = EDAReport(output_dir=base_dir / 'reports')
    eda_report.generate(df, target_column='condition')

    X, y = processor.prepare_features(df)
    X_transformed = processor.transform_features(X)

    trainer = HeartDiseaseModelTrainer(output_dir=base_dir / 'models')
    results_df = trainer.train_and_compare(X_transformed, y)
    logger.info('Model comparison results:\n%s', results_df.to_string(index=False))

    trainer.save_artifacts(processor.preprocessor)
    logger.info('Saved best model and preprocessing pipeline to %s', base_dir / 'models')


if __name__ == '__main__':
    main()
