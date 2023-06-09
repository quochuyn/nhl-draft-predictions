# train_draft_position_predictor.py

import json
import pickle

import numpy as np
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler, OneHotEncoder, FunctionTransformer



# errors from incorrect formatting of input to TfidfVectorizer was resolved with
# https://stackoverflow.com/questions/26367075/countvectorizer-attributeerror-numpy-ndarray-object-has-no-attribute-lower
def _get_text_data(x):
    return x.ravel()

def train(prospect_df):
    r"""
    Train and test a model to predict the draft position. The idea to organize
    the pipeline into preprocessing the features based on their data type comes
    from the course SIADS 696.

    Parameters
    ----------
    prospect_df : pandas.DataFrame
        The NHL prospects dataframe ready to be vectorized and
        trained on the model.
    
    Returns
    -------
    model :
        The trained scikit-learn model.
    data : dict
        A dictionary holding the train and test data for the model.
    """

    X = prospect_df[['all_reports', 'Height', 'Weight', 'Position']]
    y = prospect_df['Drafted']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.33, random_state=42
    )

    numeric_cols = ['Height', 'Weight']
    numeric_transformer = Pipeline(
        steps=[
            ('imputer', SimpleImputer(strategy='median')),
            ('scalar', StandardScaler())
        ]
    )

    categorical_cols = ['Position']
    categorical_transformer = Pipeline(
        steps=[
            # encode position with OneHotEncoder over LabelEncoder
            #   since LabelEncoder defines an unintentional ordering
            #   (e.g., 0 < 1 < 2)
            ('imputer', SimpleImputer(strategy='constant', fill_value='')),
            ('encoder', OneHotEncoder())
        ]
    )
    
    text_cols = ['all_reports']
    text_transformer = Pipeline(
        steps=[
            ('imputer', SimpleImputer(strategy='constant', fill_value='')),
            ('selector', FunctionTransformer(_get_text_data)),
            ('vectorizer', TfidfVectorizer(analyzer='word', ngram_range=(1,2)))
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
            ('clf', LogisticRegression())
        ]
    )

    model.fit(X_train, y_train)

    # Grid Search Parameters for LogisticRegression
    param_grid = {
        'clf__penalty' : ['l1', 'l2'],
        'clf__C' : np.logspace(-4, 4, 20),
        'clf__solver' : ['liblinear']
    }

    # Training config
    kfold = StratifiedKFold(n_splits=3)
    scoring = {'Accuracy': 'accuracy', 'F1': 'f1_macro'}
    refit = 'F1'

    # Perform GridSearch
    rf_model = GridSearchCV(
        model, 
        param_grid=param_grid, 
        cv=kfold, 
        scoring=scoring, 
        refit=refit, 
        n_jobs=-1, 
        return_train_score=True, 
        verbose=1
    )
    rf_model.fit(X_train, y_train)
    rf_best = rf_model.best_estimator_

    # best found for now to be 0.16 
    # without draft position (I think there is bias with that data)
    metrics = {
        # TODO: output the parameters of the best estimator
        'score' : rf_best.score(X_test, y_test)
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

    with open(args.output_metrics, 'w') as write_file:
        json.dump(metrics, write_file)
