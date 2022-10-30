import pandas as pd
import numpy as np


credits = pd.read_csv('credits.csv',sep=',',engine="python",
                 error_bad_lines=False,  
                 encoding='utf-8')
movies = pd.read_csv('movies.csv')

# # pd.set_option('display.max_columns', None)
# # pd.set_option('display.max_rows', None)

# # print(credits)

# # movies


movies = movies.merge(credits, on='title')

# # movies.info()

# # """Removing some extra columns"""

movies = movies[['id', 'title', 'genres', 'overview', 'keywords', 'cast', 'crew']]

# # """Data Preprocessing"""

# # CHECKING MISSING VALUES

# # movies.isna().sum()

# # Rows That contain NaN values

# # movies[movies['overview'].isna()]

# # Removing Null Values

movies.dropna(inplace=True)

# # CHECKING DUPLICATE ROWS

# # movies.duplicated().sum()

import ast

# # The ast module helps Python applications to process trees of the Python abstract syntax grammar.

# # movies.iloc[0].genres

# # ast.literal_eval(movies.iloc[0].genres)[0]['name']

def genre_extract(genre_list):
  genre_name = []
  for i in ast.literal_eval(genre_list):
    genre_name.append(i['name'])
  return genre_name

movies['genres'] = movies['genres'].apply(genre_extract)

# # same case in "keywords" column
movies['keywords'] = movies['keywords'].apply(genre_extract)

# # dropping some more columns

movies.drop(columns=['id', 'cast', 'crew'], inplace=True)

movies['overview'] = movies['overview'].apply(lambda x: x.split())

# # removing spaces b/w words

movies['overview'] = movies['overview'].apply(lambda x: [i.replace(' ', '') for i in x])
movies['genres'] = movies['genres'].apply(lambda x: [i.replace(' ', '') for i in x])
movies['keywords'] = movies['keywords'].apply(lambda x: [i.replace(' ', '') for i in x])

# # Creating a new column tags
# #combination of all the columns

movies['tags'] = movies['overview']+ movies['genres']+movies['keywords']

# print(movies.columns)

# # removing some columns
# # creating new DataFrame

df = movies[['title', 'tags']]


# # converting to list

df['tags'] = df['tags'].apply(lambda x: ' '.join(x))


# """TEXT PREPROCESSING"""

df['tags'] = df['tags'].apply(lambda x: x.lower())


# # """Using **Count Vectorizer**

# # (Used to convert a collection of text documents to a matrix of token counts)
# # """

from sklearn.feature_extraction.text import CountVectorizer

cv = CountVectorizer(max_features=5000, stop_words='english')

vectors = cv.fit_transform(df['tags']).toarray()


# # """Importing Natural Language Toolkit (A nlp library)"""

# # import nltk
from nltk.stem.porter import PorterStemmer

# # Stemmer is used to convert word into its root word

ps = PorterStemmer()

def stemming(text):
  t = []
  for i in text.split():
    t.append(ps.stem(i))
  return ' '.join(t)

# # df['tags'].head()

df['tags'] = df['tags'].apply(stemming)

# # df['tags'].head()

# # """Using **Cosing Similarity**

# # (Cosine similarity is a metric used to measure how similar the documents are irrespective of their size)
# # """

from sklearn.metrics.pairwise import cosine_similarity

similarity = cosine_similarity(vectors)

# # similarity[0]

# # sorted(list(enumerate(similarity[0])), reverse=True, key=lambda x: x[1])[:5]

def recommendation_function(movie):
  movie_index = df[df['title']==movie].index[0]
  distances = similarity[movie_index]
  movie_list = sorted(list(enumerate(distances)), reverse=True, key=lambda x: x[1])[1:5]
  pkp = []
  # print("\n\nRecommended Movies: \n")
  for i in movie_list:
    pkp.append(df.iloc[i[0]].title)
  return pkp  

# recommendation_function('Obvious Child')

# print(recommendation_function('Avatar'))