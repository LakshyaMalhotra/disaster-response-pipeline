import json
import plotly
import pandas as pd

from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

from flask import Flask
from flask import render_template, request, jsonify
from plotly.graph_objs import Bar
from sklearn.externals import joblib
from sqlalchemy import create_engine


app = Flask(__name__)

def tokenize(text):
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens

# load data
engine = create_engine('sqlite:///../data/DisasterResponse.db')
df = pd.read_sql_table('message_categories', engine)

basic_amenities = (df['water'] == 1) | (df['food'] == 1) | (df['shelter'] == 1)
natural_disasters = ['weather_related', 'floods', 'storm', 'fire', 
                     'earthquake', 'cold', 'other_weather']
need_aid = ['aid_related', 'medical_help', 'medical_products', 'other_aid']

# filter the data of interest from the dataframe
disasters = pd.DataFrame(df[natural_disasters].sum(), columns=['counts'])

top_10_basic_amenities = pd.DataFrame((df[basic_amenities][df.columns[4:]]
                                       .sum()
                                       .sort_values(ascending=False)[3:10]),
                                      columns=['counts'])

messages_aid_related = pd.DataFrame(df[need_aid].sum(), columns=['counts'])


# load model
model = joblib.load("../models/classifier.pkl")


# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():
    
    # extract data needed for visuals
    # TODO: Below is an example - modify to extract data for your own visuals
#     genre_counts = df.groupby('genre').count()['message']
#     genre_names = list(genre_counts.index)
    disasters_names = disasters.index
    disasters_counts = disasters['counts']
    
    top_amenities_names = top_10_basic_amenities.index
    top_amenities_counts = top_10_basic_amenities['counts']
    
    aid_related_names = messages_aid_related.index
    aid_related_counts = messages_aid_related['counts'] 
    
    # create visuals
    # TODO: Below is an example - modify to create your own visuals
    graphs = [
        {
            'data': [
                Bar(
                    x=disasters_names,
                    y=disasters_counts
                )
            ],

            'layout': {
                'title': 'Number of responses for each natural disaster',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Type of natural disaster"
                }
            }
        },
        {
            'data':[
                Bar(
                    x=top_amenities_names,
                    y=top_amenities_counts
                )
            ],
            
            'layout': {
                'title': 'Top categories where respondents need basic amenities',
                'yaxis':{
                    'title': "Count"
                },
                'xaxis':{
                    'title': "Categories"
                }
            }
        },
        {
            'data':[
                Bar(
                    x=aid_related_names,
                    y=aid_related_counts
                )
            ],
            
            'layout': {
                'title': 'Categories where respondents need some aid',
                'yaxis':{
                    'title': "Count"
                },
                'xaxis':{
                    'title': "Categories"
                }
            }
        }
    ]
    
    # encode plotly graphs in JSON
    ids = ["graph-{}".format(i) for i, _ in enumerate(graphs)]
    graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)
    
    # render web page with plotly graphs
    return render_template('master.html', ids=ids, graphJSON=graphJSON)


# web page that handles user query and displays model results
@app.route('/go')
def go():
    # save user input in query
    query = request.args.get('query', '') 

    # use model to predict classification for query
    classification_labels = model.predict([query])[0]
    classification_results = dict(zip(df.columns[4:], classification_labels))

    # This will render the go.html Please see that file. 
    return render_template(
        'go.html',
        query=query,
        classification_result=classification_results
    )


def main():
    app.run(host='0.0.0.0', port=3001, debug=True)


if __name__ == '__main__':
    main()