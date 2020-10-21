import sys
import os
import numpy as np
import pandas as pd
from sqlalchemy import create_engine


def load_data(messages_filepath, categories_filepath):
    """
    Read the CSV files and load the data into pandas dataframe after merging the data 
    from both files.

    Arguments:
    ==========
    :param messages_filepath: (str) Path of the CSV file containing messages.
    :param categories_filepath: (str) Path of the CSV file containing categories.

    :return: Merged dataframe containing both messages and categories data.
    """
    # Reading the data from both files and storing them in dataframes
    df1 = pd.read_csv(messages_filepath)
    df2 = pd.read_csv(categories_filepath)

    # Merging the dataframes into one
    df = pd.merge(df1, df2, on="id")

    return df


def clean_data(df):
    """
    Cleans the categorical columns of the dataframe. Splits categories of the
    dataframe into separate category columns.

    Arguments:
    ==========
    :param df: (Pandas dataframe) Dataframe to be cleaned.

    :return: Cleaned dataframe.
    """

    # Splitting the categories column at ';' and creating a new dataframe of 36
    # individual category columns
    categories = df["categories"].str.split(";", expand=True)

    # Select the first row of categories dataframe as the column names
    column_names = list(categories.iloc[0])

    # Cleaning the column names
    column_names = [column.split("-")[0] for column in column_names]

    # Renaming the columns of categories dataframe
    categories.columns = column_names

    # Convert category values to just numbers 0 or 1
    categories[categories.columns] = categories[categories.columns].apply(
        lambda x: x.str.split("-").str[-1].astype(int)
    )

    # Replace category columns in `df` with new category columns
    df.drop("categories", axis=1, inplace=True)
    df = pd.concat([df, categories], axis=1)

    # Remove duplicates
    df.drop_duplicates(inplace=True)

    return df


def save_data(df, database_filename):
    """
    Saves the clean dataset into an sqlite database. It uses pandas `to_sql` 
    method and `SQLAlchemy` library.

    Arguments:
    ==========
    :param df: (pandas dataframe) dataframe containing cleaned data.
    :param database_path: (str) path to the sql database.
    :param table_name: (str) name of the sql table.

    :return: None
    """
    # Creating an sqlalchemy engine
    arg = os.path.join("sqlite:///", database_filename)
    engine = create_engine(arg)
    df.to_sql("message_categories", engine, index=False)


def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print(
            "Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}".format(
                messages_filepath, categories_filepath
            )
        )
        df = load_data(messages_filepath, categories_filepath)

        print("Cleaning data...")
        df = clean_data(df)

        print("Saving data...\n    DATABASE: {}".format(database_filepath))
        save_data(df, database_filepath)

        print("Cleaned data saved to database!")

    else:
        print(
            "Please provide the filepaths of the messages and categories "
            "datasets as the first and second argument respectively, as "
            "well as the filepath of the database to save the cleaned data "
            "to as the third argument. \n\nExample: python process_data.py "
            "disaster_messages.csv disaster_categories.csv "
            "DisasterResponse.db"
        )


if __name__ == "__main__":
    main()
