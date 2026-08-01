import pandas as pd
from pathlib import Path

for name in ['heart.csv', 'heart_cleveland_upload.csv']:
    p = Path('dataset') / name
    df = pd.read_csv(p)
    print('FILE', name)
    print('shape', df.shape)
    print('columns', list(df.columns))
    print('dtypes')
    print(df.dtypes.astype(str).to_string())
    print('head')
    print(df.head(3).to_string(index=False))
    target = df.iloc[:, -1]
    print('target unique', sorted(pd.Series(target.dropna().unique()).astype(str).tolist()))
    print('---')
