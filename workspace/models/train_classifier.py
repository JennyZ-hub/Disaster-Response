import sys
import nltk
nltk.download(['punkt', 'wordnet', 'averaged_perceptron_tagger'])
import re
import numpy as np
import pandas as pd
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sqlalchemy import create_engine
from sklearn.pipeline import Pipeline,FeatureUnion
from sklearn.metrics import confusion_matrix,classification_report
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.utils.multiclass import type_of_target
import pickle

def load_data(database_filepath):
"""This function loads data from sql database and extract dummy variables for
different categories"""
    engine = create_engine('sqlite:///{}'.format(database_filepath))
    df = pd.read_sql_table('DisasterResponse', 'sqlite:///{}'.format(database_filepath))
    X = df.message.values
    Y = df.iloc[:, 4:]
    Y=Y.replace(2,0)
    category_names = Y.columns
    return X,Y,category_names

def tokenize(text):
""" This function Returns a tokenized copy of messages and Lemmatizer the words"""
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()
    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)
    return clean_tokens

def build_model():
""" This function builds a machine learning pipeline and train the pipeline"""
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', RandomForestClassifier())
    ])
    parameters = {
        'tfidf__use_idf': (True, False),
        'clf__n_estimators': [100, 200],
        'clf__min_samples_split': [2, 3, 4]}
    model = GridSearchCV(pipeline, param_grid=parameters,cv=2,n_jobs=-1,verbose=2)
    return model

def evaluate_model(model, X_test, Y_test, category_names):
""" This function tests the model"""
    Y_pred = model.predict(X_test)
    print(classification_report(Y_test.values, Y_pred,target_names=category_names))


def save_model(model, model_filepath):
""" This function saves the model to pickle file"""
    pickle.dump(model, open(model_filepath, 'wb'))


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)

        print('Building model...')
        model = build_model()

        print('Training model...')
        model.fit(X_train, Y_train)

        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

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
