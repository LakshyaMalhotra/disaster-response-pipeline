import os
import sys
import re
import nltk
import pickle

nltk.download(["punkt", "wordnet", "stopwords"])

import warnings

warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
from sqlalchemy import create_engine
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords

from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.multioutput import MultiOutputClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer


def load_data(database_filepath):
    """
    Load the data from the database file. Use pandas to read the table in the 
    database and create a dataframe.

    Arguments:
    ==========
    :param database_filepath: (database file) Path of the sqlite database file

    :return: X, Y, category names
    """
    # Load the data from database file
    arg = os.path.join("sqlite:///", database_filepath)
    engine = create_engine(arg)

    # Read the data and write in a pandas dataframe
    df = pd.read_sql_table("message_categories", engine)

    # Creating the features and labels dataframes
    X = df["message"]

    category_names = [
        column
        for column in df.columns
        if (
            (column != "id")
            & (column != "message")
            & (column != "original")
            & (column != "genre")
        )
    ]

    Y = df[category_names]

    return X, Y, category_names


def tokenize(text):
    """
    Tokenization function to process the text data.

    Arguments:
    ==========
    :param text: (str) Document of text

    :return: Tokenized text
    """
    # Getting the unique stopwords
    stop_words = set(
        stopwords.words("english")
    )  # this step is very important or gridsearch will give errors!

    # Removing the punctuations
    text = re.sub(r"[^a-zA-Z0-9]", " ", text)

    # Tokenizing the text
    tokens = word_tokenize(text)

    # Removing the stopwords
    tokens = [token for token in tokens if token not in stop_words]

    # Instantiating word lemmatizer
    lemmatizer = WordNetLemmatizer()

    # Normalizing, cleaning, and converting all the tokens to word root
    clean_tokens = list()
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens


def build_model():
    """
    Create a ML pipeline and builds the model. Uses count vectorizer and tfidf for
    NLP operations and Random forest and Multioutput classifier for classification
    of messages.

    Arguments: None

    returns: None
    """
    # Creating a sklearn pipeline with hyperparameters tuned by Grid Search
    pipeline = Pipeline(
        [
            (
                "vect",
                CountVectorizer(
                    tokenizer=tokenize, ngram_range=(1, 2), max_features=5000
                ),
            ),
            ("tfidf", TfidfTransformer(use_idf=True)),
            (
                "clf",
                MultiOutputClassifier(
                    RandomForestClassifier(
                        n_estimators=200, min_samples_split=3
                    ),
                    n_jobs=-1,
                ),
            ),
        ],
        verbose=True,
    )
    return pipeline


def evaluate_model(model, X_test, Y_test, category_names):
    """
    Evaluates the performance of the model. Calculates metrics like f1-score, 
    precision, recall for each class and store them in a dataframe.

    Arguments:
    ==========
    :param model: Model trained on training data
    :param X_test: (pandas dataframe) Test data set
    :param Y_test: (pandas dataframe) Test set labels
    :param category_names: (list) Category names / Y_test columns

    :return: None
    """
    # Calculating the predictions
    Y_pred = model.predict(X_test)
    Y_pred = pd.DataFrame(Y_pred, columns=category_names, index=Y_test.index)

    # Creating a dataframe used to hold the performance metrics for each class
    scores = pd.DataFrame(
        columns=["class_name", "f1-score", "precision", "recall", "support"]
    )

    # Calculating f1-score, precision, recall for each class
    for category in category_names:
        if category == "related":
            output = classification_report(
                Y_test[category].values,
                Y_pred[category].values,
                target_names=["False", "True", "Neither"],
                output_dict=True,  # this arg needs scikit-learn >= 0.20
            )
        elif category == "child_alone":
            output = classification_report(
                Y_test[category].values,
                Y_pred[category].values,
                target_names=["False"],
                output_dict=True,
            )
        else:
            output = classification_report(
                Y_test[category].values,
                Y_pred[category].values,
                output_dict=True,
                target_names=["False", "True"],
            )
        class_scores = pd.DataFrame(output).transpose()
        class_scores["class_name"] = category
        scores = pd.concat([scores, class_scores], axis=0)
    scores = scores.reset_index()
    scores.rename(columns={"index": "values"}, inplace=True)

    # Converting the scores to multi-index dataframe
    scores = scores.set_index(["class_name", "values"])

    # Printing out the "micro-avg" or "accuracy" for each class
    print("Accuracy/micro-avg for different classes: \n")
    print(scores.xs("accuracy", level=1)["f1-score"])


def save_model(model, model_filepath):
    """
    Saves the trained model in a file.

    Arguments:
    ==========
    :param model: Trained model
    :param model_filepath: (str) Path to the saved model file

    :return: None
    """
    with open(model_filepath, "wb") as f:
        pickle.dump(model, f)


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print("Loading data...\n    DATABASE: {}".format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)

        print("Building model...")
        model = build_model()

        print("Training model...")
        model.fit(X_train, Y_train)

        print("Evaluating model...")
        evaluate_model(model, X_test, Y_test, category_names)

        print("Saving model...\n    MODEL: {}".format(model_filepath))
        save_model(model, model_filepath)

        print("Trained model saved!")

    else:
        print(
            "Please provide the filepath of the disaster messages database "
            "as the first argument and the filepath of the pickle file to "
            "save the model to as the second argument. \n\nExample: python "
            "train_classifier.py ../data/DisasterResponse.db classifier.pkl"
        )


if __name__ == "__main__":
    main()
