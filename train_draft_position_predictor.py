# train_draft_position_predictor.py

import json
import pickle

import numpy as np
import pandas as pd

from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler, OneHotEncoder, FunctionTransformer

from sentence_transformers import SentenceTransformer



class CustomBertTransformer(BaseEstimator, TransformerMixin):

    def __init__(self):
        # throw all arguments to the superclass
        super().__init__()

        self.model = SentenceTransformer('all-mpnet-base-v2')

    def fit(self, X, y=None):
        # no fitting
        return self
    
    def transform(self, X, y=None):
        # return bert embeddings
        embeddings = self.model.encode(X)
        return embeddings
        

# errors from incorrect formatting of input to TfidfVectorizer was resolved with
# https://stackoverflow.com/questions/26367075/countvectorizer-attributeerror-numpy-ndarray-object-has-no-attribute-lower
def _get_text_data(x):
    return x.ravel()

def train(prospect_df, numeric_cols=None, categorical_cols=None, text_cols=None):
    r"""
    Train and test a model to predict the draft position. The idea to organize
    the pipeline into preprocessing the features based on their data type comes
    from the course SIADS 696.

    Parameters
    ----------
    prospect_df : pandas.DataFrame
        The NHL prospects dataframe ready to be vectorized and
        trained on the model.
    numeric_cols : list, default=None
        The numeric columns/features of X.
    categorical_cols : list, default=None
        The categorical columns/features of X, but not including the text columns.
    text_cols : list, default=None
        The text columns/features of X.
    
    Returns
    -------
    model :
        The trained scikit-learn model.
    data : dict
        A dictionary holding the train and test data for the model.
    """

    if numeric_cols is None:
        numeric_cols = ['Height', 'Weight']
    if categorical_cols is None:
        categorical_cols = ['Position']
    if text_cols is None:
        text_cols = ['all_reports']

    X = prospect_df[numeric_cols + categorical_cols + text_cols]
    y = prospect_df['Drafted']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.33, random_state=42
    )

    numeric_transformer = Pipeline(
        steps=[
            ('imputer', SimpleImputer(strategy='median')),
            ('scalar', StandardScaler())
        ]
    )

    categorical_transformer = Pipeline(
        steps=[
            # encode position with OneHotEncoder over LabelEncoder
            #   since LabelEncoder defines an unintentional ordering
            #   (e.g., 0 < 1 < 2)
            ('imputer', SimpleImputer(strategy='constant', fill_value='')),
            ('encoder', OneHotEncoder())
        ]
    )
    
    text_transformer = Pipeline(
        steps=[
            ('imputer', SimpleImputer(strategy='constant', fill_value='')),
            ('selector', FunctionTransformer(_get_text_data)),
            # ('vectorizer', TfidfVectorizer(analyzer='word', ngram_range=(1,2)))
            ('vectorizer', CustomBertTransformer())
        ]
    )

    feature_transformer = ColumnTransformer(
        transformers=[
            ('numeric', numeric_transformer, numeric_cols),
            ('categorical', categorical_transformer, categorical_cols),
            ('text', text_transformer, text_cols)
        ]
    )

    model = Pipeline(
        steps=[
            ('features', feature_transformer),
            ('clf', LinearRegression())
        ]
    )

    model.fit(X_train, y_train)

    # best found for now to be 0.16 
    # without draft position (I think there is bias with that data)
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    metrics = {
        **model.best_params_,
        'train_score' : model.score(X_train, y_train),
        'test_score' : model.score(X_test, y_test),
        'mae' : mean_absolute_error(y_test, y_test_pred),
        'mse' : mean_squared_error(y_test, y_test_pred),
        'r2' : r2_score(y_test, y_test_pred),
    }

    return model, metrics



if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        'input_file', help='the preprocessed NHL prospects data file (.CSV)'
    )
    parser.add_argument(
        'output_model', help='the trained model (.PKL)'
    )
    parser.add_argument(
        'output_metrics', help='the metrics (.JSON)',
    )
    parser.add_argument(
        '-v', '--verbose', action='store_true',
        help='display metrics',
    )
    args = parser.parse_args()

    preprocessed_df = pd.read_csv(args.input_file)
    model, metrics = train(preprocessed_df)

    if args.verbose:
        print(json.dumps(metrics, indent=4))

    with open(args.output_model, 'wb+') as write_file:
        pickle.dump(model, write_file)

    with open(args.output_metrics, 'a+') as write_file:
        write_file.write(json.dumps(metrics, indent=4) + '\n')
