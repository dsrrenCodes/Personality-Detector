import numpy as np
import pandas as pd
import sklearn  
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import FunctionTransformer
from sklearn.preprocessing import OneHotEncoder

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import loguniform,randint
from sklearn.model_selection import train_test_split
from sklearn.ensemble import StackingClassifier
import joblib




class CascadingImputer(BaseEstimator,TransformerMixin):

    def __init__(self,bin_columns=None,impute_steps=None):
        '''
        bin_columns : list , columns to create bins
        impute_steps: list of tuples , [(target_col_to_fill_nan_values,[grp_col1,grp_col2,...])]
        '''
        self.bin_columns= bin_columns or []
        self.impute_steps = impute_steps or []
        
    def fit(self,X,y=None):
        return self

    def transform(self,X):
        df=X.copy()
        for col in self.bin_columns:
            if col in df.columns:
                # Check if all values are the same
                if df[col].nunique() == 1:
                    # If all values are the same, create a single bin
                    df[f'{col}_bin'] = 'Q1'
                else:
                    # Use qcut with duplicates='drop' to handle cases where bin edges would be the same
                    try:
                        df[f'{col}_bin'] = pd.qcut(df[col], 4, labels=['Q1','Q2','Q3','Q4'], duplicates='drop')
                    except ValueError:
                        # Fallback to fewer bins if qcut with 4 bins fails
                        n_bins = min(4, df[col].nunique())
                        df[f'{col}_bin'] = pd.qcut(df[col], n_bins, labels=[f'Q{i+1}' for i in range(n_bins)], duplicates='drop')

        for target_col,grp_cols in self.impute_steps:
            for grp_col in grp_cols:
                grp_col=grp_col+'_bin'
                df[target_col]=df[target_col].fillna(df.groupby(grp_col)[target_col].transform('median'))
                
        for col in self.bin_columns:
            df.drop(columns=[f'{col}_bin'],inplace=True)
            
        return df




def fill_stage_drained_with_unKnow(X):
    df = X.copy()
    
    if 'Stage_fear' in df.columns:
        df['Stage_fear'] = df['Stage_fear'].fillna('UnKnow')
    
    if 'Drained_after_socializing' in df.columns:
        df['Drained_after_socializing'] = df['Drained_after_socializing'].fillna('UnKnow')
    
    return df


def strip_prefix(params, prefix):
    """Remove pipeline prefix from parameter names"""
    stripped_params = {}
    prefix_with_separator = f"{prefix}__"
    
    for key, value in params.items():
        if key.startswith(prefix_with_separator):
            # Remove the prefix and separator
            new_key = key[len(prefix_with_separator):]
            stripped_params[new_key] = value
        else:
          
            stripped_params[key] = value
    
    return stripped_params








def hyperparameter_search(model, param_grid, X_train, y_train, X_test, y_test,preprocessing=None, model_name='model', n_iter=20, cv=3,scoring='accuracy', verbose=1):
    
    """
    returns best estimator from random search and results [dict]
    """
    
    #create pipeline
    if preprocessing is not None:
        pipeline = Pipeline([
            ('preprocessing', preprocessing),
            (model_name, model)
        ])
        #add correct prefix to param_grid keys if not already present
        formatted_param_grid = {}
        expected_prefix = f'{model_name}__'
        
        for key, value in param_grid.items():
            if not key.startswith(expected_prefix):
                formatted_param_grid[f'{expected_prefix}{key}'] = value
            else:
                formatted_param_grid[key] = value
    else:
        pipeline = model
        formatted_param_grid = param_grid


    rs = RandomizedSearchCV(pipeline, 
        formatted_param_grid, 
        n_iter=n_iter, 
        cv=cv, 
        scoring=scoring,
        verbose=verbose)
    
    rs.fit(X_train, y_train)
    
   
    train_score = rs.score(X_train, y_train)
    test_score = rs.score(X_test, y_test)
    y_pred = rs.predict(X_test)
    
   
    results = {
        'best_estimator': rs.best_estimator_,
        'best_params': rs.best_params_,
        'best_cv_score': rs.best_score_,
        'train_score': train_score,
        'test_score': test_score,
        'y_pred': y_pred,
        'randomized_search': rs
    }
    print(f'Model: {model_name}')
    print(f"Best CV Score: {rs.best_score_:.4f}")
    print(f"Test Score: {test_score:.4f}")
    print(f"Best Parameters: {rs.best_params_}")
    print('\n')
    #rs.best_estimator returns the entire pipeline including preprocessing step so lets just return the model only
    return rs.best_params_,results


import os
# Use a relative path to the CSV file
current_dir = os.path.dirname(os.path.abspath(__file__))
train_df = pd.read_csv(os.path.join(current_dir, 'train_df.csv'))

train_df.drop(columns=['Unnamed: 0'],inplace=True)

casading_imputer = CascadingImputer(
    bin_columns=['Social_event_attendance', 'Going_outside',
                 'Friends_circle_size', 'Post_frequency'],
    impute_steps=[
        ('Time_spent_Alone',       ['Social_event_attendance', 'Going_outside']),
        ('Social_event_attendance',['Going_outside', 'Friends_circle_size', 'Post_frequency']),
        ('Going_outside',          ['Social_event_attendance', 'Friends_circle_size', 'Post_frequency']),
        ('Friends_circle_size',    ['Social_event_attendance', 'Going_outside', 'Post_frequency']),
        ('Post_frequency',         ['Going_outside', 'Social_event_attendance','Friends_circle_size'])
    ]
)

cat_pipe = Pipeline([
    ('fill_unknown', FunctionTransformer(fill_stage_drained_with_unKnow)),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

num_pipe= Pipeline(
    [('fill_nan',casading_imputer),
    ('ss',StandardScaler())]
)

preprocessing=ColumnTransformer(
    [('nan_imputer_num_and_others',num_pipe,['Time_spent_Alone','Social_event_attendance','Going_outside','Friends_circle_size','Post_frequency']),
     ('nan_imputer_cat_and_others',cat_pipe,['Stage_fear','Drained_after_socializing'])
    ], remainder='passthrough'
)



X=train_df.drop(columns=['Personality'])
y=train_df['Personality']


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42,stratify=y)

rf_clean_params= {'bootstrap': False,
 'max_depth': 7,
 'max_features': 'log2',
 'min_samples_leaf': 3,
 'min_samples_split': 2,
 'n_estimators': 201}

svc_clean_params={'C': 16.242649850626332,
 'degree': 2,
 'gamma': 0.0006186205226242456,
 'kernel': 'rbf',
 'probability': True}

lg_clean_params={'C': 0.06163397863898331,
 'max_iter': 1000,
 'penalty': 'l2',
 'solver': 'liblinear'}


final_model=StackingClassifier([('random_forest',RandomForestClassifier(**rf_clean_params)),('svc',SVC(**svc_clean_params))],final_estimator=LogisticRegression(**lg_clean_params))
final_model_pipeline=Pipeline([('pp',preprocessing),('final_model',final_model)])

if __name__ == "__main__":
    print('training model...')
    final_model_pipeline.fit(X_train,y_train)
    print('model trained successfully!')
    joblib.dump(final_model_pipeline, 'model_pipeline.pkl')
    print('model saved')