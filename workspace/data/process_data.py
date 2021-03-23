import sys
import pandas as pd
import sqlite3
from sqlalchemy import create_engine

def load_data(messages_filepath, categories_filepath):
"""Load messages and categories csv files
   Description:
   This function loads disaster_messages and disaster_categories files and merge them on 'id'
   and save as df. And, create dummy variables for different categories.
"""

    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    df = pd.merge(messages,categories,on='id')
    categories = categories['categories'].str.split(";",expand=True)
    row = categories.loc[0]
    fun = lambda x:x[:-2]
    category_colnames = row.apply(fun)
    categories.columns = category_colnames
    for column in categories:
        f=lambda x: x[-1]
        categories[column] = categories[column].apply(f)
        categories[column] = pd.to_numeric(categories[column])
    df = df.drop('categories', axis=1)
    df = pd.concat([df,categories], axis=1)
    return df


def clean_data(df):
"""This function clean DataFrame df."""

    df=df.drop_duplicates('message')
    return df

def save_data(df, database_filename):
"""This function save df to sql database."""

   engine = create_engine('sqlite:///{}'.format(database_filename))
   df.to_sql('DisasterResponse', engine, index=False)

def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)

        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)

        print('Cleaned data saved to database!')

    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()
