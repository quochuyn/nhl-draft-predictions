# clean_nhl_prospects.py

import numpy as np
import pandas as pd
from sklearn import preprocessing



def clean(input_file='data/prospect-data.csv', raw=False):
    r"""
    Load and clean the NHL prospects data which is a collection 
    of 2014-2022 NHL player scouting reports from various public 
    sports news outlets (e.g., The Athletic, EP Rinkside, ESPN).

    Parameters
    ----------
    input_file : str, default='data/prospect-data.csv'
        The file path for the NHL prospects data. 
    raw : bool, default=False
        Boolean value whether to return the raw data.
    
    Returns
    -------
    df : pandas.DataFrame
        The output data frame of the NHL prospects data.
    """

    df = pd.read_csv(input_file)
    
    # no cleaning
    if raw:
        return df

    # encode position
    pos_le = preprocessing.LabelEncoder()
    df.loc[:,'position_encoded'] = pos_le.fit_transform(df['Position'])

    # encode team names
    team_le = preprocessing.LabelEncoder()
    df.loc[:,'team_encoded'] = team_le.fit_transform(df['Team'])

    return df



if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument(
        'input_file', help='the raw NHL prospects data file (.CSV)'
    )
    parser.add_argument(
        'output_file', help='the clean NHL prospects data file (.CSV)'
    )
    args = parser.parse_args()

    clean_df = clean(args.input_file)
    clean_df.to_csv(args.output_file, index=False)
