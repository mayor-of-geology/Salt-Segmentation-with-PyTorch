
from warnings import filterwarnings
from pandas import read_csv, DataFrame
from sklearn.model_selection import train_test_split
from typing import Tuple
from pathlib import Path

filterwarnings('ignore')

def data(ROOT:Path) -> Tuple[DataFrame, DataFrame]:
    
    #getting data

    train_df = read_csv(ROOT/'train.csv', index_col='id', usecols=[0])
    depths_df = read_csv(ROOT/'depths.csv', index_col='id')
    train_df = train_df.join(depths_df)
    valid_df = depths_df[~depths_df.index.isin(train_df.index)]

    train_df['images'] = [ROOT/f'train/images/{idx}.png' for idx in train_df.index]
    train_df['masks'] = [ROOT/f'train/masks/{idx}.png' for idx in train_df.index]

    #splitting into train and test data
    trainset, testset = train_test_split(train_df, test_size=0.2, random_state=32)
    
    return trainset, testset
