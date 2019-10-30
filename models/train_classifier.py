# import libraries
import pickle
import re
import sys
import numpy as np
import pandas as pd
from sqlalchemy import create_engine

import nltk
nltk.download(['punkt', 'wordnet'])
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize, RegexpTokenizer

from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer


def load_data(database_filepath):
    ''' Load data, transform it into a dataframe, 
    and return the text and labels as X and Y.
    Params
    ======
        database_filepath (str): filepath to SQLite database
    '''
    engine = create_engine('sqlite:///'+database_filepath)
    df = pd.read_sql_table('DisasterResponse', con=engine)
    X = df['message']
    y = df.iloc[:, 4:]
    
    return X, y


def tokenize(text):
    ''' Tokenize and clean the text.
    Params
    ======
        text (str): original message data
    '''
    # Normalize text
    text = re.sub(r'[^a-zA-Z0-9]', ' ', text.lower())
    # Tokenize text
    tokenizer = RegexpTokenizer(r'\w+')
    tokens = tokenizer.tokenize(text)
    # Initiate lemmatizer
    lemmatizer = WordNetLemmatizer()
    # Iterate through each token
    clean_tokens = []
    for tok in tokens:
        # Lemmatize, normalize case, and remove leading/trailing white space
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens    


def build_model():
    ''' Make a machine learning pipeline with term frequency-inverse document
    frequency, random forest classification, and grid search for tuning parameters.
    '''
    # Build pipeline
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(RandomForestClassifier(random_state=42)))])
    # Specify parameters for grid search
    parameters = {
        'vect__max_df': (0.5, 0.75, 1.0),
        'clf__estimator__criterion': ['entropy', 'gini'],
        'clf__estimator__max_depth': [None, 5, 10],
        'clf__estimator__max_features': ['auto', 'log2']}
    # Create grid search object
    cv = GridSearchCV(pipeline, param_grid=parameters, cv=3, n_jobs=3)
    
    return cv


def evaluate_model(model, X_test, y_test):
    ''' Provide an evaluation for the machine learning model performance.
    Params
    ======
        model (str): machine learning pipeline object
        X_test (str): test features
        y_test (str): test labels
    '''
    # Predict test data
    y_pred = model.predict(X_test)
    # Calculate the overall accuracy of the model
    accuracy = (y_pred == y_test).mean().mean()
    print('OVERALL ACCURACY: {0:.2f}% \n'.format(accuracy*100))
    # Make a dataframe of the results
    results = pd.DataFrame(y_pred, columns=y_test.columns)
    # Display f1 score, precision, and recall for each disaster output category
    for column in y_test.columns:
        print('CATEGORY: {}\n'.format(column))
        print(classification_report(y_test[column], results[column]))


def save_model(model, model_filepath):
    ''' Save the trained machine learning model as a Pickle file. 
    Params
    ======
    model (str): machine learning pipeline object
    model_filepath (str): destination path to the Pickle file
    '''
    pickle.dump(model, open(model_filepath, 'wb'))


def main():
    ''' Extract data from SQLite database, train a machine learning classifier model
    on the data, evaluate model performance, and save the trained model.
    '''
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, y = load_data(database_filepath)
        # Perform train-test split
        X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, y_test)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()
